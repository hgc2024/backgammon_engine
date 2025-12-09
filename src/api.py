from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional, Any
import numpy as np
import torch
import os
import copy

from src.game import BackgammonGame, GamePhase
from src.search import ExpectiminimaxAgent

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State (MVP: Single Session)
game = BackgammonGame()
agent = None # Agent
history_stack = [] # Undo stack

# Models
class MoveRequest(BaseModel):
    action_idx: int

class PartialMoveRequest(BaseModel):
    move: Tuple[int, int or str] # Start, End ('off' or int)

class GameState(BaseModel):
    board: List[int]
    bar: List[int]
    off: List[int]
    turn: int
    dice: List[int]
    cube_value: int
    cube_owner: int
    legal_moves: List[Any] # Complex structure
    phase: str
    winner: int
    score: List[int]
    
# Helpers
def get_state_dict():
    # Convert numpy/enums to JSON friendly
    return {
        "board": game.board.tolist(),
        "bar": game.bar,
        "off": game.off,
        "turn": game.turn,
        "dice": game.dice,
        "cube_value": game.cube_value,
        "cube_owner": game.cube_owner,
        "legal_moves": game.legal_moves, # List of lists of tuples
        "phase": game.phase.name,
        "winner": -1 if game.phase != GamePhase.GAME_OVER else (0 if game.score[0] > game.score[1] else 1), # Logic check needed
        "score": game.score
    }

def save_undo_snapshot():
    global history_stack
    # Deep copy game state
    snapshot = copy.deepcopy(game)
    history_stack.append(snapshot)
    if len(history_stack) > 10: history_stack.pop(0)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Backgammon Engine API"}

@app.post("/start")
def start_game():
    global history_stack
    game.reset_match()
    history_stack = []
    return get_state_dict()

@app.get("/gamestate")
def get_gamestate():
    return get_state_dict()

@app.post("/roll")
def roll_dice():
    if game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
        save_undo_snapshot()
        game.step(0) # Roll
    return get_state_dict()

@app.get("/moves")
def get_legal_moves():
    """Returns list of atomic (from, to) legal moves for current dice."""
    if game.phase == GamePhase.DECIDE_MOVE:
        moves = game.get_legal_partial_moves()
        # Convert ('off') to specific code if needed, but JSON handles mixed types check frontend
        # Types: List[Tuple[int, int|str]]
        # Start is always int (0-23 or bar). End is int or 'off'.
        return moves
    return []

@app.post("/step")
def play_partial_move(req: PartialMoveRequest):
    save_undo_snapshot()
    
    # Clean input: req.move might have 'off' string
    start, end = req.move
    
    # Validate Phase
    if game.phase != GamePhase.DECIDE_MOVE:
        return {"error": "Not in Move Phase"}
        
    try:
        # Step Partial returns (points, winner, game_over)
        game.step_partial((start, end)) # Ignores return reward for now
        
        # Check if turn is done? 
        # step_partial handles `turn` switch?
        # game.step_partial logic:
        # "4. Checks if turn is finished."
        # So if dice empty, it switches turn.
        
        return get_state_dict()
    except Exception as e:
        print(f"Move Error: {e}")
        return {"error": str(e)}

# Agent Config
MODEL_PATH = "checkpoints/gen2_champion.pth"
if os.path.exists(MODEL_PATH):
    agent = ExpectiminimaxAgent(MODEL_PATH, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded Agent: {MODEL_PATH}")

@app.post("/ai-move")
def play_ai_move():
    if not agent:
        return {"error": "No Agent Loaded"}
    
    # 1. Roll if needed
    if game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
        game.step(0)
        
    # 2. Move
    if game.phase == GamePhase.DECIDE_MOVE:
        action = agent.get_action(game, depth=2)
        if action is not None:
            game.step(action)
            
    return get_state_dict()

@app.post("/undo")
def undo_turn():
    global game, history_stack
    if history_stack:
        game = history_stack.pop()
        return get_state_dict()
    return {"error": "Nothing to undo"}
    

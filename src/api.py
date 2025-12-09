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
class StartRequest(BaseModel):
    first_player: int # 0=White, 1=Red, -1=Random

class AIMoveRequest(BaseModel):
    depth: int = 2

# Track logs
move_history: List[str] = []

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
        "legal_moves": game.legal_moves, 
        "phase": game.phase.name,
        "winner": -1 if game.phase != GamePhase.GAME_OVER else (0 if game.score[0] > game.score[1] else 1),
        "score": game.score,
        "history": move_history
    }

def log_move(msg: str):
    move_history.append(msg)
    if len(move_history) > 50: move_history.pop(0)

def save_undo_snapshot():
    global history_stack
    # Deep copy game state AND history
    snapshot = (copy.deepcopy(game), copy.deepcopy(move_history))
    history_stack.append(snapshot)
    if len(history_stack) > 10: history_stack.pop(0)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Backgammon Engine API"}

@app.post("/start")
def start_game(req: Optional[StartRequest] = None):
    global history_stack, move_history
    game.reset_match()
    history_stack = []
    move_history = ["Game Started"]
    
    # Handle custom start
    if req and req.first_player != -1:
        # Force turn. reset_match usually randomizes or sets 0?
        # game.reset_match() -> sets random turn.
        # We override.
        game.turn = req.first_player
        
    return get_state_dict()

@app.get("/gamestate")
def get_gamestate():
    return get_state_dict()

@app.post("/roll")
def roll_dice():
    if game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
        save_undo_snapshot()
        game.step(0) # Roll
        log_move(f"P{game.turn} Rolled {game.dice}")
    return get_state_dict()

@app.get("/moves")
def get_legal_moves():
    """Returns list of atomic (from, to) legal moves for current dice."""
    if game.phase == GamePhase.DECIDE_MOVE:
        moves = game.get_legal_partial_moves()
        return moves
    return []

@app.post("/step")
def play_partial_move(req: PartialMoveRequest):
    save_undo_snapshot()
    start, end = req.move
    if game.phase != GamePhase.DECIDE_MOVE:
        return {"error": "Not in Move Phase"}
    try:
        game.step_partial((start, end)) 
        log_move(f"P{game.turn} Moved {start}->{end}")
        return get_state_dict()
    except Exception as e:
        print(f"Move Error: {e}")
        return {"error": str(e)}

# Agent Config
MODEL_PATH = "best_so_far_gen2.pth"
if os.path.exists(MODEL_PATH):
    agent = ExpectiminimaxAgent(MODEL_PATH, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded Agent: {MODEL_PATH}")

@app.post("/ai-move")
def play_ai_move(req: Optional[AIMoveRequest] = None):
    if not agent:
        return {"error": "No Agent Loaded"}
    
    depth = req.depth if req else 1 # Default 1 ply (Fast) if not specified, or user pref?
    
    # 1. Roll if needed
    if game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
        game.step(0)
        log_move(f"CPU Rolled {game.dice}")
        
    # 2. Move
    if game.phase == GamePhase.DECIDE_MOVE:
        action = agent.get_action(game, depth=depth)
        if action is not None:
            # Decode move for logging
            move_seq = game.legal_moves[action] # List[Tuple[int, int]]
            move_str = ", ".join([f"{start}->{end}" for start, end in move_seq])
            
            game.step(action)
            log_move(f"CPU: {move_str}")
            
    return get_state_dict()

@app.post("/undo")
def undo_turn():
    global game, history_stack, move_history
    if history_stack:
        snapshot = history_stack.pop()
        game = snapshot[0]
        move_history = snapshot[1]
        log_move("Undo")
        return get_state_dict()
    return {"error": "Nothing to undo"}

@app.post("/pass")
def pass_turn():
    """Allows user to manually end turn IF no moves are possible."""
    # Check if legal moves exist
    moves = game.get_legal_partial_moves()
    if not moves:
        # Switch Turn
        game.turn = 1 - game.turn
        game.phase = GamePhase.DECIDE_CUBE_OR_ROLL
        log_move(f"P{1-game.turn} Passed (No Moves)")
        return get_state_dict()
    else:
        return {"error": "You still have legal moves!", "legal_moves": moves}
    

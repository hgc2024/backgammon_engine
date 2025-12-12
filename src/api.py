from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional, Any
import numpy as np
import torch
import os
import copy
import random

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
    move: Tuple[int | str, int | str] # Start (int/'bar'), End (int/'off')

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
    reset_score: bool = True # Default to True (Reset Match) unless specified

class AIMoveRequest(BaseModel):
    depth: int = 2
    style: str = "aggressive"

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
        "pips": [int(x) for x in game.get_pip_counts()], 
        "device": str(agent.device) if agent else "N/A",
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
    
    # Defaults
    should_reset_score = True
    first_player = -1
    
    if req:
        should_reset_score = req.reset_score
        first_player = req.first_player
    
    if should_reset_score:
        game.reset_match()
        move_history = ["Match Reset. New Game Started."]
    else:
        game.reset_game()
        move_history = ["New Game Started (Score Kept)."]
        
    history_stack = []
    
    # Handle custom start
    if first_player != -1:
        game.turn = first_player
    else:
        # Random Start
        game.turn = random.randint(0, 1)
            
    print(f"Start Game Requested(P{first_player}, Reset={should_reset_score}) -> Actual Turn: {game.turn}")
    return get_state_dict()

@app.get("/gamestate")
def get_gamestate():
    return get_state_dict()

@app.post("/roll")
def roll_dice():
    if game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
        # User requested: Undo should NOT undo the roll, only moves.
        # So we clear the stack here to prevent undoing past this point.
        global history_stack
        history_stack = []
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
MODEL_PATH = "best_so_far_gen5.pth"
    
if os.path.exists(MODEL_PATH):
    # Agent will auto-detect Gen 5 vs Gen 4 based on checkpoint keys in search.py
    agent = ExpectiminimaxAgent(MODEL_PATH, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded Agent: {MODEL_PATH}")

@app.post("/ai-move")
def play_ai_move(req: Optional[AIMoveRequest] = None):
    if not agent:
        return {"error": "No Agent Loaded"}
    
    depth = 3
    style = "aggressive"
    
    # 1. Roll if needed
    if game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
        game.step(0)
        log_move(f"CPU Rolled {game.dice}")
        
    # 2. Move
    if game.phase == GamePhase.DECIDE_MOVE:
        # Check if we are blocked
        if not game.legal_moves:
             game.step(0) # Logic ignores index if empty and switches turn
             log_move(f"CPU: No moves (Pass)")
             return get_state_dict()             
    
        action = agent.get_action(game, depth=depth, style=style)
        if action is not None:
            # Decode move for logging
            move_seq = game.legal_moves[action] # List[Tuple[int, int]]
            move_str = ", ".join([f"{start}->{end}" for start, end in move_seq])
            
            # Get Equity (Gen 4)
            val = getattr(agent, "last_value", 0.0)
            
            # Rough Win % Estimate: (Eq + 1) / 2
            # Clamped between 0% and 100%
            win_est = (val + 1.0) / 2.0
            win_est = max(0.0, min(1.0, win_est)) * 100
            
            game.step(action)
            log_move(f"CPU: {move_str} (Eq: {val:.3f}, Win: ~{int(win_est)}%)")
            
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
    

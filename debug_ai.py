import torch
import numpy as np
from src.game import BackgammonGame, GamePhase
from src.search import ExpectiminimaxAgent
from src.train_td import get_obs_from_state # Verify imports
import sys

def debug_ai():
    MODEL_PATH = "best_so_far_gen2.pth"
    print(f"Loading {MODEL_PATH}...")
    try:
        agent = ExpectiminimaxAgent(MODEL_PATH, device="cpu")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    game = BackgammonGame()
    
    # 1. Test Hit Scenario
    print("\n--- TEST 1: Obvious Hit Opportunity ---")
    game.reset_match()
    game.turn = 0 # Agent Turn
    # Clear board
    game.board = np.zeros(24, dtype=int)
    # Agent at 24 (Index 23)
    game.board[23] = 2
    # Opponent Blot at 18 (Index 17)
    game.board[17] = -1 
    # Opponent Stack at 0
    game.board[0] = -5
    
    # Dice = 6, 5 (Non-double)
    roll = (6, 5)
    game.dice = list(roll)
    
    print(f"Board State: Agent at 23(2), Opponent Blot at 17(-1). Dice={roll}")
    
    # Verify Hit:
    # 23->17 (Uses 6) HITS.
    # Then 23->18 (Uses 5).
    # OR 23->18 (5), 18->12 (6).
    
    full_moves = game.get_legal_moves(roll)
    game.legal_moves = full_moves # Set for Agent
    
    # print(f"Legal Moves: {full_moves}") # Too noisy
    
    # Analyze Values
    if not full_moves:
        print("No moves?")
    else:
        # Replicate _run_1ply logic
        boards = []
        for seq in full_moves:
            b, ba, o = game.get_afterstate(seq)
            boards.append((b, ba, o))
            
        # Eval
        opponent = 1 - game.turn # 1
        vals = agent._evaluate_states(boards, opponent, opponent)
        
        print("Move Evaluations:")
        for i, seq in enumerate(full_moves):
            val = vals[i].item()
            # print(f"Move {seq}: Opponent Value = {val:.4f}")
            
        best_idx = torch.argmin(vals).item()
        print(f"AI Chose: {full_moves[best_idx]} (OppVal {vals[best_idx]:.4f}) [Minimizing]")
        
        # Heuristic Check
        # Does the chosen move hit?
        chosen_move = full_moves[best_idx]
        hits = False
        # Check if 17 appears as destination
        for m in chosen_move:
            if m[1] == 17: hits = True
        print(f"Does it hit? {hits}")

    # 2. Test Bear Off vs Safety
    print("\n--- TEST 2: Bear Off Race ---")
    game.board = np.zeros(24, dtype=int)
    # Agent at 0, 1 (Home)
    game.board[0] = 2
    game.board[1] = 2
    # Opponent far away
    game.board[23] = -2
    
    game.dice = [1, 2] 
    
    full_moves = game.get_legal_moves(game.dice)
    print(f"Num Legal Moves: {len(full_moves)}")
    
    boards = []
    for seq in full_moves:
        boards.append(game.get_afterstate(seq))
    vals = agent._evaluate_states(boards, 1, 1)
    
    best_idx = torch.argmin(vals).item()
    print(f"AI Chose: {full_moves[best_idx]} (OppVal {vals[best_idx]:.4f})")
    
    # Check if bear off
    bears_off = False
    for m in full_moves[best_idx]:
        if m[1] == 'off': bears_off = True
    print(f"Bears off? {bears_off}")

if __name__ == "__main__":
    debug_ai()

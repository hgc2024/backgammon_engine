import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.game import BackgammonGame
from src.search import ExpectiminimaxAgent
import time

def play_game(agent0, agent1):
    game = BackgammonGame()
    
    # print(f"Game Start. P0 (White): {'Heuristic' if agent0.use_race_heuristic else 'Baseline'}. P1 (Red): {'Heuristic' if agent1.use_race_heuristic else 'Baseline'}")
    
    steps = 0
    winner = -1
    while winner == -1:
        steps += 1
        current_agent = agent0 if game.turn == 0 else agent1
        
        # Roll
        # Need to ensure dice are rolled if needed.
        # Check game state. If game._phase != PLAY (e.g. roll needed)
        # But game.roll_dice() forces roll?
        # self.roll_dice() sets self.dice.
        
        if not game.dice:
             d1, d2 = game.roll_dice()
             game.dice = [d1, d2]
             if d1 == d2:
                 game.dice = [d1, d1, d1, d1]
             game.legal_moves = game.get_legal_moves(game.dice)
             
        if steps < 5:
             print(f"Step {steps} Turn{game.turn} Dice{game.dice} Moves{len(game.legal_moves)}")
        
        # Get Action (1-Ply for speed)
        best_idx = current_agent.get_action(game, depth=1)
        
        # Apply
        if best_idx is not None:
            pts, w, done = game.step(best_idx)
            if done:
                return w
        else:
            # Pass
            game.turn = 1 - game.turn
            game.dice = [] # Clear dice after pass? Or does step_action do it?
            # step_action usually clears dice. If manual pass, we must clear.
            # But get_action returns None ONLY if no moves. 
            # We should probably call step_action logic for pass?
            # step_action doesn't handle pass if moves is empty?
            # Check step_action logic. step_action uses index.
            # If moves empty, usually we don't call step_action?
            # Manual turn switch.
            pass
            
        if steps > 2000:
             print("Stalemate aborted.")
             return -1
             
    return -1

def run_series(n_games=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "best_so_far_gen5.pth"
    
    print(f"Loading Model: {model_path} on {device}")
    
    # Initialize Agents
    # Agent 1: With Heuristic
    agent_heur = ExpectiminimaxAgent(model_path, device=device, use_race_heuristic=True)
    
    # Agent 2: Without Heuristic
    agent_base = ExpectiminimaxAgent(model_path, device=device, use_race_heuristic=False)
    
    heur_wins = 0
    base_wins = 0
    
    start_time = time.time()
    
    print(f"Starting {n_games} Games Series (2-Ply)...")
    
    for i in range(n_games):
        # Swap sides every game to be fair
        if i % 2 == 0:
            # P0: Heuristic, P1: Baseline
            winner = play_game(agent_heur, agent_base)
            if winner == 0:
                heur_wins += 1
                # print(f"Game {i+1}: Heuristic (White) Won")
            elif winner == 1:
                base_wins += 1
                # print(f"Game {i+1}: Baseline (Red) Won")
        else:
            # P0: Baseline, P1: Heuristic
            winner = play_game(agent_base, agent_heur)
            if winner == 0:
                base_wins += 1
                # print(f"Game {i+1}: Baseline (White) Won")
            elif winner == 1:
                heur_wins += 1
                # print(f"Game {i+1}: Heuristic (Red) Won")
        
            print(f"Progress: {i+1}/{n_games}. Heuristic Wins: {heur_wins} - Baseline Wins: {base_wins}", flush=True)
            
    elapsed = time.time() - start_time
    print(f"\n--- Results ---", flush=True)
    print(f"Total Games: {n_games}", flush=True)
    print(f"Heuristic Bot Wins: {heur_wins} ({heur_wins/n_games*100:.1f}%)", flush=True)
    print(f"Baseline Bot Wins:  {base_wins} ({base_wins/n_games*100:.1f}%)", flush=True)
    print(f"Time Taken: {elapsed:.1f}s", flush=True)

if __name__ == "__main__":
    run_series(6)

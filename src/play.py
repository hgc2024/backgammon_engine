import sys
import os
import numpy as np
from src.game import BackgammonGame, GamePhase
from src.env import BackgammonEnv
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

def main():
    print("Welcome to Backgammon Engine Play Mode!")
    print("Loading model...")
    model_path = "backgammon_final"
    model = None
    if os.path.exists(model_path + ".zip"):
        model = MaskablePPO.load(model_path)
        print("Model loaded.")
    else:
        print("No trained model found. Playing against Random Agent.")

    env = BackgammonEnv()
    obs, info = env.reset()
    game = env.game
    
    # Human plays Player 0 (White). Agent plays Player 1 (Black).
    
    while True:
        print("\n" + "="*20)
        print(game.render_ascii())
        
        if game.phase == GamePhase.GAME_OVER:
            print(f"Game Over! Score: {game.score}")
            break
            
        if game.turn == 0:
            # Human Turn
            print("\nYour Turn!")
            
            if game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
                print("Actions: [r] Roll, [d] Double")
                choice = input("Choice: ").strip().lower()
                if choice == 'd':
                    action = 1
                else:
                    action = 0
            
            elif game.phase == GamePhase.RESPOND_TO_DOUBLE:
                print("Opponent Doubled! Actions: [t] Take, [d] Drop")
                choice = input("Choice: ").strip().lower()
                if choice == 'd':
                    action = 1 # Drop in our Env mapping? 
                    # Env: DECIDE_CUBE_OR_ROLL map: 0=Roll, 1=Double
                    # Env: RESPOND: 0=Take, 1=Drop
                    action = 1 
                else:
                    action = 0
                    
            elif game.phase == GamePhase.DECIDE_MOVE:
                print(f"Roll: {game.current_roll}")
                moves = game.legal_moves
                if not moves:
                    print("No moves available. Press Enter.")
                    input()
                    # We still need to step the env to process the "pass"
                    # Env masking handles this? 
                    # If no moves, game logic usually auto-switches.
                    # Wait, our `_roll_and_start_turn` handles blocked.
                    # But if we are in DECIDE_MOVE, moves exist.
                    pass
                else:
                    print("Available Moves:")
                    for idx, seq in enumerate(moves):
                        print(f"{idx}: {seq}")
                    
                    choice = -1
                    while choice < 0 or choice >= len(moves):
                        try:
                            choice = int(input("Select Move Index: "))
                        except:
                            pass
                    action = choice
            
            # Step Env
            # Note: Env expects action index depending on phase
            # But wait, env.step(action) needs generic action index.
            # Our Env maps phase to action meaning.
            # 0/1 for decide/respond. 0..N for moves.
            
            obs, reward, done, truncated, info = env.step(action)
            
            if done:
                print(f"Game Finished. Reward: {reward}")
                game.reset_game() # Next game in match?
                # Actually env.reset() or just continue loop?
                # If match not over, continue.
                if max(game.score) >= game.match_target:
                    print("Match Over!")
                    break
        else:
            # Agent Turn
            print("\nAgent Thinking...")
            
            # Masking
            action_masks = env.action_masks()
            
            if model:
                action, _ = model.predict(obs, action_masks=action_masks)
            else:
                # Random valid
                valid_indices = np.where(action_masks)[0]
                action = np.random.choice(valid_indices)
                
            print(f"Agent chose action: {action}")
            obs, reward, done, truncated, info = env.step(action)
            
            if done:
                print(f"Game Finished. Reward: {reward}")
                if max(game.score) >= game.match_target:
                    print("Match Over!")
                    break

if __name__ == "__main__":
    main()

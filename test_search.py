from src.game import BackgammonGame
from src.search import ExpectiminimaxAgent
import torch

def test():
    print("Initializing Game...")
    game = BackgammonGame()
    
    # Create dummy model file if not exists
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU specific test")
        device = "cpu"
    else:
        device = "cuda"

    print(f"Device: {device}")
    
    # We need a model file. Use the one from training if exists, else create dummy.
    # Assuming 'td_backgammon.pth' might exist from previous turn.
    # If not, create one.
    try:
        agent = ExpectiminimaxAgent("td_backgammon.pth", device=device)
    except:
        print("Model file not found. Creating dummy.")
        from src.model import BackgammonValueNet
        net = BackgammonValueNet()
        torch.save(net.state_dict(), "td_backgammon.pth")
        agent = ExpectiminimaxAgent("td_backgammon.pth", device=device)
        
    print("Agent Loaded.")
    
    # Force a state with moves
    game.reset_game()
    game.step(0) # Roll
    print(f"Rolled: {game.current_roll}")
    print(f"Moves: {len(game.legal_moves)}")
    
    # Test 1-Ply
    print("Testing 1-Ply...")
    action = agent.get_action(game, depth=1)
    print(f"1-Ply Action: {action}")
    
    # Test 2-Ply (Expectiminimax)
    # WARNING: Slow if legal moves is high.
    # Limit moves for test?
    if len(game.legal_moves) > 0:
        print("Testing 2-Ply (Expectiminimax)...")
        # Crop moves for speed test if needed, but let's try full
        action2 = agent.get_action(game, depth=2)
        print(f"2-Ply Action: {action2}")
        
    print("Test Complete.")

if __name__ == "__main__":
    test()

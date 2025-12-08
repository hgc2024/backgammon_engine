import torch
import os

def inspect(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        # Load on CPU to avoid CUDA dependency for inspection
        checkpoint = torch.load(path, map_location="cpu")
        
        print(f"--- Inspecting: {path} ---")
        if isinstance(checkpoint, dict):
            print("Type: Dictionary Checkpoint")
            print("Keys:", list(checkpoint.keys()))
            
            if 'episode' in checkpoint:
                print(f"Episode: {checkpoint['episode']}")
            if 'best_win_rate' in checkpoint:
                print(f"Best Win Rate: {checkpoint['best_win_rate'] * 100:.2f}%")
            if 'model_state_dict' in checkpoint:
                print("Model State: Present")
            
            # Legacy fallback: current code saves minimal dict for checkpoints/
            # But td_backgammon_best.pth has full dict.
            # Let's see what's in best_so_far.pth (It was copied from best.pth? or just minimal?)
            # Code says:
            # torch.save({ 'episode': ..., 'model_state_dict': ... }, "checkpoints/best_so_far.pth")
            # So it might NOT have 'best_win_rate' key explicitly depending on which block saved it.
            
        else:
            print("Type: Legacy Weights Only (No Metadata)")
            
    except Exception as e:
        print(f"Error loading: {e}")

if __name__ == "__main__":
    inspect("checkpoints/best_so_far.pth")
    print("\n")
    inspect("td_backgammon_best.pth")


import torch
import numpy as np
from src.search import ExpectiminimaxAgent
from unittest.mock import MagicMock

def test_evaluate_single():
    print("Testing _evaluate_single...")
    
    # Mock the Agent to avoid loading heavy models
    # We can't easily mock __init__ if we instantiate directly, 
    # but we can subclass or just monkeypatch.
    
    # Simpler: Instantiate with a dummy file? No, torch.load will fail.
    # Let's mock torch.load
    
    original_load = torch.load
    
    try:
        # Create a dummy checkpoint
        dummy_ckpt = {
            'model_state_dict': {} 
        }
        torch.load = MagicMock(return_value=dummy_ckpt)
        
        # We also need to mock the Network class so it doesn't fail to load state dict
        # The code imports BackgammonValueNet from src.model or src.model_gen5
        
        # Let's just patch the network creation in the class?
        # ExpectiminimaxAgent uses `src.model.BackgammonValueNet` if not gen5.
        
        # Actually, let's just Try-Catch the init, or assume we have a valid model path.
        # The user has "td_backgammon.pth" in root.
        model_path = "c:\\Users\\henry-cao-local\\OneDrive\\Desktop\\Self_Learning\\ML_Projects\\backgammon_engine\\td_backgammon.pth"
        
        # NOTE: Loading the real model is better for integration test, but might be slow or fail if env is wrong.
        # Let's try real load.
        
        torch.load = original_load # Restore
        
        agent = ExpectiminimaxAgent(model_path, device='cpu')
        print("Agent loaded.")
        
        # Prepare dummy inputs
        b = np.zeros(24, dtype=np.float32)
        ba = np.zeros(2, dtype=np.float32)
        o = np.zeros(2, dtype=np.float32)
        p = 0
        
        # Call the method
        val = agent._evaluate_single(b, ba, o, p, style="aggressive")
        
        print(f"Result: {val}", flush=True)
        
        if isinstance(val, float):
             print("SUCCESS: Method returned a float.", flush=True)
        else:
             print(f"FAILURE: Method returned {type(val)}", flush=True)
             
    except Exception as e:
        print(f"ERROR: {e}", flush=True)

if __name__ == "__main__":
    test_evaluate_single()

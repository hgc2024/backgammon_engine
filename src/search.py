import torch
import numpy as np
import itertools
from src.game import BackgammonGame
from src.train_td import get_obs_from_state as get_obs_gen4
from src.model_gen5 import BackgammonValueNetGen5

def get_obs_gen5(board, bar, off, turn, score, cube, player_perspective):
    # 1. Base Gen 4 Features (198)
    # [Player0(24*4), Player1(24*4), Bar(2), Off(2), Turn(2), Cube(1?)]
    # Actually train_td.get_obs_from_state returns 198.
    # Gen 5 adds Match Score (2). Total 200.
    
    # We can reuse gen4 obs and append match info.
    # get_obs_gen4 (from train_td) signature: (board, bar, off, perspective_player, score, cube_val, turn)
    # We pass dummy score/cube for the base calculation if not used, or real ones if consistent.
    # Gen 4 model didn't use score, but the function requires it.
    base_obs_np = get_obs_gen4(board, bar, off, turn, score, cube, 1) # Returns np.array
    base_obs = base_obs_np.tolist() # Convert to list to append
    
    # 2. Match Scores (Normalized by Match Target if possible, or just raw/15?)
    # Train Script logic: user provided scores.
    # Here game.score is [p0, p1].
    # Feature 1: My Score / 25.0
    # Feature 2: Opp Score / 25.0
    # (Assuming match to 25 or similar? Train used /25.0 normalization)
    
    # Perspective Check
    if player_perspective == 0:
        s_my = score[0]
        s_opp = score[1]
    else:
        s_my = score[1]
        s_opp = score[0]
        
    base_obs.append(s_my / 25.0)
    base_obs.append(s_opp / 25.0)
    
    return np.array(base_obs, dtype=np.float32)

class ExpectiminimaxAgent:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.is_gen5 = False
        
        # Load Checkpoint First to Determine Architecture
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint

        # Heuristic to detect Gen 5: Check for transformer keys or known Gen 5 input size layer
        # Gen 5 has 'transformer.layers...'
        is_gen5_ckpt = any('transformer' in k for k in state_dict.keys())
        
        if is_gen5_ckpt:
            print(f">>> Loading Gen 5 Model from {model_path}")
            self.net = BackgammonValueNetGen5().to(device)
            self.is_gen5 = True
        else:
            print(f">>> Loading Gen 4 Model from {model_path}")
            from src.model import BackgammonValueNet
            self.net = BackgammonValueNet().to(device)
            
        self.net.load_state_dict(state_dict)
        self.net.eval()
        
        # Precompute all dice probabilities for 2-ply
        # 36 total rolls.
        self.dice_dist = {}
        for d1 in range(1, 7):
            for d2 in range(1, 7):
                roll = tuple(sorted((d1, d2)))
                if roll not in self.dice_dist:
                    self.dice_dist[roll] = 0
                self.dice_dist[roll] += 1
        
        # Normalize
        for k in self.dice_dist:
            self.dice_dist[k] /= 36.0
            
        self.last_value = 0.0

    def get_action(self, game, roll=None, depth=1, style="aggressive"):
        """
        Returns the best action index from game.legal_moves.
        Depth 1 = Greedy.
        Depth 2 = Star-Minimax.
        Style = 'aggressive' (Money Play) or 'safe' (Risk Averse).
        """
        moves = game.legal_moves
        if not moves:
            return None
            
        if depth == 1:
            return self._run_1ply(game, moves, style)
        elif depth >= 2:
            return self._run_2ply(game, moves, style)

    def _evaluate_states(self, boards, player, perspective_player, style="aggressive", current_score=None):
        """
        Evaluates a batch of board states.
        Returns tensor of values from `perspective_player`'s point of view.
        """
        obs_list = []
        scores = current_score if current_score else [0, 0]
        
        for (b, ba, o) in boards:
             if self.is_gen5:
                 # Gen 5 needs Score Info
                 obs = get_obs_gen5(b, ba, o, perspective_player, scores, 1, perspective_player)
             else:
                 # Gen 4 needs dummy args for signature match
                 # board, bar, off, perspective_player, score, cube_val, turn
                 obs = get_obs_gen4(b, ba, o, perspective_player, [0,0], 1, 1)
             obs_list.append(obs)
             
        batch = torch.tensor(np.array(obs_list), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits, _ = self.net(batch) # Ignore Pip Head for value search
            probs = torch.softmax(logits, dim=1)
            
            # Equity Calculation based on Style
            if style == "safe":
                 weights = torch.tensor([-4.0, -3.0, -2.0, 1.0, 1.1, 1.2], device=self.device)
            else:
                 weights = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], device=self.device)
                 
            values = torch.sum(probs * weights, dim=1) # [N]
            
        return values

    def _run_1ply(self, game, moves, style="aggressive"):
        boards = []
        for seq in moves:
            boards.append(game.get_afterstate(seq)) # (b, ba, o)
            
        opponent = 1 - game.turn
        # Pass game.score to evaluation
        values = self._evaluate_states(boards, opponent, opponent, style, current_score=game.score)
        
        # values is P(Opponent Wins). We want to MINIMIZE Opponent Advantage.
        best_val_for_opp = torch.min(values).item()
        best_idx = torch.argmin(values).item()
        
        self.last_value = -1.0 * best_val_for_opp
        return best_idx

    def _run_2ply(self, game, moves, style="aggressive"):
        # --- PRUNING STEP (STAR-MINIMAX) ---
        # 1. Run 1-Ply eval on ALL moves to identify candidates.
        boards_1ply = []
        for seq in moves:
            boards_1ply.append(game.get_afterstate(seq))
            
        opponent_1ply = 1 - game.turn
        values_1ply = self._evaluate_states(boards_1ply, opponent_1ply, opponent_1ply, style, current_score=game.score)
        
        # Zip moves with their 1-ply values
        scored_moves = []
        for i, val in enumerate(values_1ply.cpu().numpy()):
            scored_moves.append((val, i, moves[i]))
            
        # Sort by Value (Ascending) -> Minimize Opponent Equity
        scored_moves.sort(key=lambda x: x[0])
        
        # Select Top K Candidates
        TOP_K = 5
        candidates = scored_moves[:TOP_K]
        
        # --- FULL 2-PLY SEARCH ON CANDIDATES ---
        best_expectation = -float('inf')
        best_idx_global = candidates[0][1] # Default to best 1-ply
        
        sim_game = BackgammonGame()
        current_turn = game.turn
        opponent = 1 - current_turn
        
        for (val_1ply, original_idx, seq) in candidates:
            b1, ba1, o1 = game.get_afterstate(seq)
            expected_opp_win = 0.0
            
            # Simulate Opponent Rolls
            for roll, prob in self.dice_dist.items():
                sim_game.board = b1.copy()
                sim_game.bar = ba1.copy()
                sim_game.off = o1.copy()
                sim_game.turn = opponent
                sim_game.score = game.score # Inherit score
                
                opp_moves = sim_game.get_legal_moves(roll)
                
                if not opp_moves:
                    val = self._evaluate_single(b1, ba1, o1, opponent, style, current_score=game.score)
                    expected_opp_win += val * prob
                    continue
                
                # Opponent Best Response (1-Ply for them)
                s2_boards = []
                for om in opp_moves:
                    s2_boards.append(sim_game.get_afterstate(om))
                    
                vals_s2 = self._evaluate_states(s2_boards, current_turn, current_turn, style, current_score=game.score)
                
                # Opponent chooses move that MINIMIZES my equity
                best_s2_val_for_me = torch.min(vals_s2).item()
                
                expected_opp_win += best_s2_val_for_me * prob
                
            # We want to MAXIMIZE the expected equity
            if expected_opp_win > best_expectation:
                best_expectation = expected_opp_win
                best_idx_global = original_idx
                
        self.last_value = best_expectation
        return best_idx_global

    def _evaluate_single(self, b, ba, o, p, style="aggressive", current_score=None):
        scores = current_score if current_score else [0, 0]
        if self.is_gen5:
             obs = get_obs_gen5(b, ba, o, p, scores, 1, p)
        else:
             obs = get_obs_gen4(b, ba, o, p, [0,0], 1, 1)
             
        t = torch.tensor(obs[None, :], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits, _ = self.net(t)
            probs = torch.softmax(logits, dim=1)
            
            if style == "safe":
                 weights = torch.tensor([-4.0, -3.0, -2.0, 1.0, 1.1, 1.2], device=self.device)
            else:
                 weights = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], device=self.device)
                 
            values = torch.sum(probs * weights).item()
        return v

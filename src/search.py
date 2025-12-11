import torch
import numpy as np
import itertools
from src.game import BackgammonGame
from src.train_td import get_obs_from_state

class ExpectiminimaxAgent:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        # Load Model
        from src.model import BackgammonValueNet
        self.net = BackgammonValueNet().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.net.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.net.load_state_dict(checkpoint)
        self.net.eval()
        
        # Precompute all dice probabilities for 2-ply
        # 36 total rolls.
        # 1-1, 2-2, ... 6-6 (Target Prob 1/36)
        # 1-2, 2-1 (Target Prob 2/36)
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

    def get_action(self, game, roll=None, depth=1):
        """
        Returns the best action index from game.legal_moves.
        Depth 1 = Greedy (Max 1-ply value).
        Depth 2 = Expectiminimax (Max Average of Min 2-ply value).
        """
        moves = game.legal_moves
        if not moves:
            return None
            
        if depth == 1:
            return self._run_1ply(game, moves)
        elif depth >= 2:
            return self._run_2ply(game, moves)

    def _evaluate_states(self, boards, player, perspective_player):
        """
        Evaluates a batch of board states.
        Returns tensor of values from `perspective_player`'s point of view.
        """
        obs_list = []
        for (b, ba, o) in boards:
             obs = get_obs_from_state(b, ba, o, perspective_player, [0,0], 1, perspective_player)
             obs_list.append(obs)
             
        batch = torch.tensor(np.array(obs_list), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits, _ = self.net(batch) # Ignore Pip Head for value search
            probs = torch.softmax(logits, dim=1)
            
            # Equity Calculation:
            # Classes: 0:LoseBG(-3), 1:LoseG(-2), 2:Lose(-1), 3:Win(1), 4:WinG(2), 5:WinBG(3)
            weights = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], device=self.device)
            values = torch.sum(probs * weights, dim=1) # [N]
            
            # Normalize to -1..1 range for backward compatibility?
            # Standard Equity is -3..3.
            # Minimax works fine with any range as long as consistent.
            # BUT: The UI displays "Win Est: 58%".
            # If values > 1, UI might break or look weird.
            # Ideally UI shows "Equity: +1.5".
            # For now, let's keep it raw Equity.
            
        return values

    def _run_1ply(self, game, moves):
        boards = []
        for seq in moves:
            boards.append(game.get_afterstate(seq)) # (b, ba, o)
            
        opponent = 1 - game.turn
        values = self._evaluate_states(boards, opponent, opponent)
        
        # values is P(Opponent Wins) or Opponent Advantage (-1 to 1).
        # We pick the move that minimizes this.
        best_val_for_opp = torch.min(values).item()
        best_idx = torch.argmin(values).item()
        
        # My Advantage = -1 * Best Opponent Advantage
        self.last_value = -1.0 * best_val_for_opp
        
        return best_idx

    def _run_2ply(self, game, moves):
        # --- PRUNING STEP (STAR-MINIMAX) ---
        # 1. Run 1-Ply eval on ALL moves to identify candidates.
        boards_1ply = []
        for seq in moves:
            boards_1ply.append(game.get_afterstate(seq))
            
        opponent_1ply = 1 - game.turn
        # values_1ply = Opponent's Equity. We want to MINIMIZE this.
        values_1ply = self._evaluate_states(boards_1ply, opponent_1ply, opponent_1ply)
        
        # Zip moves with their 1-ply values and indices
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
        best_idx_global = candidates[0][1] # Default to best 1-ply if something goes wrong
        
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
                
                opp_moves = sim_game.get_legal_moves(roll)
                
                if not opp_moves:
                    # Terminal or Stuck: Static Eval
                    val = self._evaluate_single(b1, ba1, o1, opponent)
                    expected_opp_win += val * prob
                    continue
                
                # Opponent Best Response (1-Ply for them)
                s2_boards = []
                for om in opp_moves:
                    s2_boards.append(sim_game.get_afterstate(om))
                    
                vals_s2 = self._evaluate_states(s2_boards, current_turn, current_turn)
                
                # Opponent chooses move that MINIMIZES my equity (vals_s2)
                # vals_s2 = My Equity.
                # Minimax: Opponent minimizes My Equity.
                best_s2_val_for_me = torch.min(vals_s2).item()
                
                expected_opp_win += best_s2_val_for_me * prob
                
            # We want to MAXIMIZE the expected equity after opponent moves
            # Note: expected_opp_win here accumulates 'best_s2_val_for_me' (My Equity).
            #So we maximize it.
            if expected_opp_win > best_expectation:
                best_expectation = expected_opp_win
                best_idx_global = original_idx
                
        self.last_value = best_expectation
        return best_idx_global

    def _evaluate_single(self, b, ba, o, p):
        # Helper for single eval
        obs = get_obs_from_state(b, ba, o, p, [0,0], 1, p)
        t = torch.tensor(obs[None, :], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits, _ = self.net(t)
            probs = torch.softmax(logits, dim=1)
            weights = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], device=self.device)
            v = torch.sum(probs * weights).item()
        return v
        return v

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
        # Construct observations
        # Context: "boards" are the raw board states (np arrays)
        # We need other state info (bar, off) which we assume is passed or attached?
        # A limitation of `get_afterstate` in game.py is it returns (board, bar, off).
        # We need to process that tuple.
        
        for (b, ba, o) in boards:
             # get_obs_from_state(board, bar, off, perspective_player, score, cube, turn)
             # Score/Cube we take from current game state (assuming no change in 1 ply)
             # Turn: The state is "After" move, so it is Opponent's turn?
             # But the Network evaluates "Current Position".
             # If we feed it as "Player X to move", the network outputs P(X Wins).
             
             obs = get_obs_from_state(b, ba, o, perspective_player, [0,0], 1, perspective_player)
             obs_list.append(obs)
             
        batch = torch.tensor(np.array(obs_list), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            values = self.net(batch).squeeze(1) # [N]
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
        best_expectation = -float('inf')
        best_idx = 0
        
        sim_game = BackgammonGame()
        current_board = game.board.copy()
        current_bar = game.bar.copy()
        current_off = game.off.copy()
        current_turn = game.turn
        opponent = 1 - current_turn
        
        for i, seq in enumerate(moves):
            b1, ba1, o1 = game.get_afterstate(seq)
            expected_opp_win = 0.0
            
            for roll, prob in self.dice_dist.items():
                sim_game.board = b1.copy()
                sim_game.bar = ba1.copy()
                sim_game.off = o1.copy()
                sim_game.turn = opponent
                
                opp_moves = sim_game.get_legal_moves(roll)
                
                if not opp_moves:
                    val = self._evaluate_single(b1, ba1, o1, opponent)
                    expected_opp_win += val * prob
                    continue
                
                s2_boards = []
                for om in opp_moves:
                    s2_boards.append(sim_game.get_afterstate(om))
                    
                vals_s2 = self._evaluate_states(s2_boards, current_turn, current_turn)
                best_s2_val_for_me = torch.min(vals_s2).item() # Opponent minimizes MY value?
                
                # WAIT. If vals_s2 evaluates "Current Turn" (My Turn), it returns My Advantage.
                # Opponent wants to MINIMIZE my advantage. Correct.
                
                expected_opp_win += best_s2_val_for_me * prob
                
            if expected_opp_win > best_expectation:
                best_expectation = expected_opp_win
                best_idx = i
                
        self.last_value = best_expectation
        return best_idx

    def _evaluate_single(self, b, ba, o, p):
        # Helper for single eval
        obs = get_obs_from_state(b, ba, o, p, [0,0], 1, p)
        t = torch.tensor(obs[None, :], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            v = self.net(t).item()
        return v

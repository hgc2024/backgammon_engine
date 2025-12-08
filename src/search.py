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
        self.net.load_state_dict(torch.load(model_path, map_location=device))
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
        """
        Greedy strategy: Pick move that leads to best state for ME.
        State Value V(s) is P(Current Player Wins).
        After I move to S', it is Opponent's turn.
        V(S') is P(Opponent Wins).
        I want to MINIMIZE V(S').
        """
        boards = []
        for seq in moves:
            boards.append(game.get_afterstate(seq)) # (b, ba, o)
            
        # We evaluate S' from OPPONENT perspective to see how good it is for them.
        opponent = 1 - game.turn
        
        # We evaluate how good the board is for the OPPONENT (who is about to move)
        # We want to minimize this value.
        values = self._evaluate_states(boards, opponent, opponent)
        
        best_idx = torch.argmin(values).item()
        return best_idx

    def _run_2ply(self, game, moves):
        """
        Lookahead 2-Ply.
        For each My Move M:
           For each Possible Dice Roll D (weighted):
               Find Opponent Best Response R given D.
               Value = V(State after R). (Opponent Wins)
           Score(M) = Weighted Avg(Value).
        
        I want to minimize Score(M). (Minimize Expected Opponent Win Prob).
        """
        
        best_expectation = -float('inf')
        best_idx = 0
        
        # Optimization: This is slow. We need a "Virtual Game" to simulate 2nd ply.
        # We can't easily clone the whole game object cheaply.
        # But we only need `get_legal_moves` and `get_afterstate`.
        # `BackgammonGame` is stateful.
        
        # Temporary instance for simulation
        sim_game = BackgammonGame()
        # We need to manually set state. 
        # BackgammonGame doesn't have `set_state`. Use internals.
        
        current_board = game.board.copy()
        current_bar = game.bar.copy()
        current_off = game.off.copy()
        current_turn = game.turn
        opponent = 1 - current_turn
        
        for i, seq in enumerate(moves):
            # 1. My Move
            b1, ba1, o1 = game.get_afterstate(seq)
            
            # Now we are at State S1. Opponent's turn.
            expected_opp_win = 0.0
            
            # 2. Iterate Dice
            for roll, prob in self.dice_dist.items():
                # Setup Sim Game at S1
                sim_game.board = b1.copy()
                sim_game.bar = ba1.copy()
                sim_game.off = o1.copy()
                sim_game.turn = opponent
                # sim_game.score/cube ignored for move logic mostly
                
                # Get Legal Moves for Opponent
                opp_moves = sim_game.get_legal_moves(roll)
                
                if not opp_moves:
                    # Opponent Passes.
                    # State remains S1. Opponent Turn Ends. My Turn.
                    # Value is V(S1) from MY perspective? 
                    # Or V(S1) from Opp perspective is Low?
                    # If I passed, board didn't change.
                    # Use V(S1) for Opponent.
                    # Actually if Opponent creates no change, I just evaluate S1.
                    val = self._evaluate_single(b1, ba1, o1, opponent)
                    expected_opp_win += val * prob
                    continue
                
                # Find Opponent's Best Response (Greedy 1-ply for them)
                # They want to Minimize MY Win Prob (Next State S2).
                # S2 is My Turn. V(S2) = P(Me Win).
                # Opponent wants to Minimize V(S2).
                
                # Generate S2 candidates
                s2_boards = []
                for om in opp_moves:
                    s2_boards.append(sim_game.get_afterstate(om))
                    
                # Evaluate S2 from MY perspective (Player 0 or 1 original)
                vals_s2 = self._evaluate_states(s2_boards, current_turn, current_turn)
                
                # Opponent chooses move that MINIMIZES my Value.
                best_s2_val_for_me = torch.min(vals_s2).item()
                
                # "Best S2 Val For Me" is my win prob.
                # Opponent Win Prob = 1 - My Win Prob?
                # Or if using Tanh (-1 to 1):
                # Opponent minimizes My Advantage.
                # If Tanh output is "My Advantage", Opponent minimizes it.
                # So the State Value resulting is `best_s2_val_for_me`.
                
                # Wait: "Score(M) = Weighted Avg(Value)"
                # If we are minimizing Opponent Win Prob, we are Maximizing My Win Prob.
                # Let's stick to Maximizing My Value.
                
                # Contribution to Expectation
                expected_opp_win += best_s2_val_for_me * prob
                
            # expected_opp_win is actually "Expected Value for ME after Opponent Response"
            # So we want to MAXIMIZE this.
            
            if expected_opp_win > best_expectation: # Initialize with -inf
                best_expectation = expected_opp_win
                best_idx = i
                
        return best_idx

    def _evaluate_single(self, b, ba, o, p):
        # Helper for single eval
        obs = get_obs_from_state(b, ba, o, p, [0,0], 1, p)
        t = torch.tensor(obs[None, :], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            v = self.net(t).item()
        return v

import torch
import numpy as np
import itertools
from src.game import BackgammonGame, get_obs_from_state as get_obs_gen4
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
    def __init__(self, model_path, device='cpu', use_race_heuristic=True):
        self.device = device
        self.model_path = model_path
        self.use_race_heuristic = use_race_heuristic
        self.is_gen5 = False
        
        # Load Checkpoint First to Determine Architecture
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            print(f"Error loading model checkpoint: {e}")
            raise
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

    def get_state_value(self, game, style="aggressive", depth=2):
        """
        Evaluates the current state.
        If depth > 0, performs lookahead (Expectiminimax).
        Returns dict with "equity" and "win_prob" (0.0-1.0).
        """
        if depth > 0:
            # LOOKAHEAD EVALUATION
            if game.dice:
                # Decision Node: Find value of Best Move
                # This populates self.last_value and self.last_win_prob
                best_idx = self.get_action(game, depth=depth)
                # If no moves (Pass), get_action returns None?
                # If None, value is evaluated from Opponent Turn perspective?
                # get_action handles pass internally? No, returns None.
                # If None, we need to eval resulting state (Turn Switch).
                
                if best_idx is None:
                    # Pass. Value is -Eval(Opponent).
                    # Approximate by evaluating state where turn is switched.
                    # Or just 0-ply eval of inverted state?
                    # get_action usually implies we are deciding.
                    # If pass, we are effectively at next state.
                    pass # TODO: Handle Pass Eval more accurately.
                    # Fallback to 0-ply if pass?
                    return self.get_state_value(game, style, depth=0)
                
                return {
                    "equity": self.last_value,
                    "win_prob": self.last_win_prob
                }
            else:
                # Chance Node: Expectation over Dice
                # This is "1-Ply Expectiminimax Eval" (Average of Best Moves)
                # or "2-Ply" if depth=2 passed to get_action?
                # User wanted 2-ply. So we pass depth=2.
                # Loop 21 rolls.
                
                total_eq = 0.0
                total_wp = 0.0
                
                # Use same dist as search
                sim_game = BackgammonGame()
                sim_game.board = game.board.copy()
                sim_game.bar = game.bar.copy()
                sim_game.off = game.off.copy()
                sim_game.turn = game.turn
                sim_game.score = game.score
                sim_game.cube_value = game.cube_value
                sim_game.cube_owner = game.cube_owner
                
                for roll, prob in self.dice_dist.items():
                    sim_game.dice = list(roll)
                    if roll[0] == roll[1]: sim_game.dice *= 2
                    
                    sim_game.legal_moves = sim_game.get_legal_moves(sim_game.dice)
                    
                    if not sim_game.legal_moves:
                        # Pass
                        # Eval resulting state (Opponent Turn)
                        # We want My Equity. = -OppEquity.
                        # Eval 0-ply for speed? Or recursive?
                        # Recursive might be too slow (21 calls).
                        # Use 0-ply for Pass cases.
                         # Opponent perspective
                        v, wp = self._evaluate_states([(sim_game.board, sim_game.bar, sim_game.off)], 1-game.turn, 1-game.turn, style, current_score=game.score)
                        
                        # My Equity = -OppEquity
                        my_eq = -v.item()
                        my_wp = 1.0 - wp.item()
                        
                        total_eq += my_eq * prob
                        total_wp += my_wp * prob
                        continue
                        
                    # Find Best Move (Depth 1 is sufficient/standard for Eval Loop)
                    # If we do Depth 2 inside loop, it's 21 * 2-ply = Slow.
                    # Depth 1 inside loop = "2-Ply Eval".
                    # i.e. Roll -> Max -> Eval.
                    # This captures "My Move" impact.
                    # User asked for 2-ply. This IS 2-ply (Chance -> Max -> Static).
                    
                    self.get_action(sim_game, depth=1) # Populates last_value
                    total_eq += self.last_value * prob
                    total_wp += self.last_win_prob * prob
                    
                return {
                    "equity": total_eq,
                    "win_prob": total_wp
                }


        # 0-PLY STATIC EVALUATION
        # Manual Obs Construction to access Probs
        p = game.turn
        scores = game.score if game.score else [0, 0]
        
        if self.is_gen5:
             obs = get_obs_gen5(game.board, game.bar, game.off, p, scores, 1, p)
        else:
             obs = get_obs_gen4(game.board, game.bar, game.off, p, [0,0], 1, 1)
             
        t = torch.tensor(obs[None, :], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits, _ = self.net(t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Equity
            if style == "safe":
                 weights = np.array([-4.0, -3.0, -2.0, 1.0, 1.1, 1.2])
            else:
                 weights = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
            
            equity = np.sum(probs * weights)
            
            # Win Prob: Sum of [Win(3), Gammon(4), Backgammon(5)]
            # Probs are [LoseBG, LoseG, Lose, Win, WinG, WinBG]
            win_prob = np.sum(probs[3:]) 
            
            return {
                "equity": float(equity),
                "win_prob": float(win_prob)
            }

    def get_action(self, game, depth=2, style="aggressive"):
        """
        Returns the best action index from game.legal_moves.
        """
        moves = game.legal_moves
        if not moves:
            return None
        if len(moves) == 1:
            return 0
            
        # --- HEURISTIC OVERRIDE FOR PURE RACES ---
        # If no contact, use simple "Max Off, Min Pips" logic.
        if self.use_race_heuristic and self._is_pure_race(game):
            # print("DEBUG: Using Race Heuristic")
            return self._run_race_heuristic(game, moves)

        if depth == 1:
            return self._run_1ply(game, moves, style)
        elif depth == 2:
            return self._run_2ply(game, moves, style)
        elif depth >= 3:
            return self._run_3ply_beam(game, moves, style)

    def _is_pure_race(self, game):
        # Check if P0 (White) and P1 (Red) have passed each other.
        b = game.board
        p0_inds = [i for i, c in enumerate(b) if c > 0]
        p1_inds = [i for i, c in enumerate(b) if c < 0]
        
        # If any on bar, not a pure race (usually) unless other has cleared?
        # Actually if on bar, you are at 24 or -1.
        if game.bar[0] > 0 or game.bar[1] > 0:
            # Technically could be race if opponent is all home, but simpler to say "No" if bar
            return False
            
        if not p0_inds or not p1_inds:
            return True # Game over essentially
            
        # P0 moves High -> Low (23->0).
        # P1 moves Low -> High (0->23).
        # Race if max(P0) < min(P1).
        return max(p0_inds) < min(p1_inds)

    def _run_race_heuristic(self, game, moves):
        best_idx = 0
        best_score = -float('inf')
        
        # Perspective
        player = game.turn
        
        for i, seq in enumerate(moves):
            # Sim
            # game.get_afterstate returns (board, bar, off)
            b, ba, o = game.get_afterstate(seq)
            
            # Score: Maximize (Off * 1000) - Pips
            # 1 checker off is worth 1000 pips. Huge.
            
            my_off = o[player]
            
            # Calc Pips
            pips = 0
            if player == 0:
                 pips += np.sum(np.maximum(b, 0) * (np.arange(24) + 1))
            else:
                 pips += np.sum(np.abs(np.minimum(b, 0)) * (24 - np.arange(24)))
            
            # Negative pips because we want to Minimize remaining.
            # Tie-Breaker: Minimize Checkers on Board (e.g. 1 checker on 2-pt > 2 checkers on 1-pt).
            # 1 checker off (10000) >> 1 pip (1) >> 1 checker count (0.01)
            
            # Count Checkers on board (excluding off)
            checkers_on_board = 15 - my_off 
            
            score = (my_off * 10000) - pips - (checkers_on_board * 0.1)
            # In pure race, safe is usually irrelevant unless rolling.
            # But minimizing blots might prevent unlucky 1-1 variance if forced to leave shot (not possible in pure race).
            # Actually, just clearing is key.
            # But "stacking" on higher points vs lower?
            # User example: 2 on 1-point vs 1 on 2-point.
            # {1:2} vs {2:1}.
            # Pips: {1:2} -> 2 pips. {2:1} -> 2 pips.
            # Off: Same.
            # Tie.
            # But {2:1} (1 checker) clears faster.
            # Add heuristic: "Minimize Checkers On Board"?
            # Equivalent to Maximize Off.
            # Add heuristic: "Minimize Max Stack Height"? No.
            # Add heuristic: "Maximize Checkers on Lower Points"?
            # i.e. move 6->5 is better than 2->1?
            # Usually reducing highest stack is good (clearing form back).
            # But user scenario: 2 checkers on 1-point vs 1 checker on 2-point.
            # Both have same pips.
            # Heuristic should prefer 1 checker on 2-point (fewer checkers left).
            # Oh, {1:2} is 2 checkers. {2:1} is 1 checker.
            # My logic earlier said {2:1} is 1 checker left.
            # Wait. {1:2} = 2 checkers at index 1.
            # {2:1} = 1 checker at index 2.
            # Maximize Off handles this!
            # {2:1} means we Bore Off 1 checker.
            # {1:2} means we Moved 2->1 (Bore Off 0).
            # So `my_off` score will totally dominate.
            
            if score > best_score:
                best_score = score
                best_idx = i
                
        return best_idx

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
             
        # Batch could be empty if no moves
        if not obs_list:
            return torch.tensor([], device=self.device)

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
            
            # --- ENDGAME AGGRESSION HEURISTIC ---
            # 1. Bear-off Bonus (Existing): Reward taking pieces off.
            off_bonuses = []
            pip_penalties = []
            
            for (b, ba, o) in boards:
                # Bonus for Off
                my_off = o[perspective_player]
                # Boost: Make bearing off HIGHLY desirable to break ties in won games.
                off_bonuses.append((my_off / 15.0) * 0.5)
                
                # Penalty for High Pips (Encourage Racing / Saving Gammon)
                # Player 0: 24 -> 0.Indices 0..23. Dist = index + 1.
                # Player 1: 0 -> 24. Indices 0..23. Dist = 24 - index.
                pips = 0
                if perspective_player == 0:
                     # Board Pips
                     # fast numpy calc: P0 are positive
                     pips += np.sum(np.maximum(b, 0) * (np.arange(24) + 1))
                     # Bar Pips (25)
                     pips += ba[0] * 25
                else:
                     # P1 are negative
                     pips += np.sum(np.abs(np.minimum(b, 0)) * (24 - np.arange(24)))
                     # Bar Pips (25)
                     pips += ba[1] * 25
                
                # Max Pips ~ 375. Penalty factor.
                # Increase penalty to ensure strict race logic when winning.
                pip_penalties.append((pips / 375.0) * 0.2)

            bonus_tensor = torch.tensor(off_bonuses, device=self.device)
            penalty_tensor = torch.tensor(pip_penalties, device=self.device)
            
            values += bonus_tensor
            values -= penalty_tensor

            # Win Probability (Sum of indices 3, 4, 5)
            # 0=LoseBG, 1=LoseG, 2=Lose, 3=Win, 4=WinG, 5=WinBG
            win_probs = torch.sum(probs[:, 3:], dim=1)

            return values, win_probs

    def _run_1ply(self, game, moves, style="aggressive"):
        boards = []
        for seq in moves:
            boards.append(game.get_afterstate(seq)) # (b, ba, o)
            
        opponent = 1 - game.turn
        # Pass game.score to evaluation
        values, win_probs = self._evaluate_states(boards, opponent, opponent, style, current_score=game.score)
        
        # values is P(Opponent Wins). We want to MINIMIZE Opponent Advantage.
        best_val_for_opp = torch.min(values).item()
        best_idx = torch.argmin(values).item()
        
        # Win Prob: best_idx corresponds to best move.
        # win_probs is from Opponent Perspective.
        # My Win Prob = 1 - Opponent Win Prob.
        opp_win_prob = win_probs[best_idx].item()
        
        self.last_value = -1.0 * best_val_for_opp
        self.last_win_prob = 1.0 - opp_win_prob # Store Win Prob
        return best_idx

    def _run_2ply(self, game, moves, style="aggressive"):
        # --- PRUNING STEP (STAR-MINIMAX) ---
        # 1. Run 1-Ply eval on ALL moves to identify candidates.
        boards_1ply = []
        for seq in moves:
            boards_1ply.append(game.get_afterstate(seq))
            
        opponent_1ply = 1 - game.turn
        values_1ply, _ = self._evaluate_states(boards_1ply, opponent_1ply, opponent_1ply, style, current_score=game.score)
        
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
        best_expectation_wp = 0.0 # Default
        best_idx_global = 0 if candidates else None
        
        sim_game = BackgammonGame()
        current_turn = game.turn
        opponent = 1 - current_turn
        
        for (val_1ply, original_idx, seq) in candidates:
            b1, ba1, o1 = game.get_afterstate(seq)
            expected_opp_win = 0.0
            expected_opp_win_prob = 0.0
            
            # Simulate Opponent Rolls
            for roll, prob in self.dice_dist.items():
                sim_game.board = b1.copy()
                sim_game.bar = ba1.copy()
                sim_game.off = o1.copy()
                sim_game.turn = opponent
                sim_game.score = game.score # Inherit score
                
                opp_moves = sim_game.get_legal_moves(roll)
                
                if not opp_moves:
                    # Use evaluate_states to get both val and win_prob
                    v, wp = self._evaluate_states([(b1, ba1, o1)], opponent, opponent, style, current_score=game.score)
                    expected_opp_win += v.item() * prob
                    expected_opp_win_prob += wp.item() * prob
                    continue
                
                # Opponent Best Response (1-Ply for them)
                s2_boards = []
                for om in opp_moves:
                    s2_boards.append(sim_game.get_afterstate(om))
                    
                vals_s2, win_probs_s2 = self._evaluate_states(s2_boards, current_turn, current_turn, style, current_score=game.score)
                
                # Opponent chooses move that MINIMIZES my equity
                best_opp_idx = torch.argmin(vals_s2).item()
                best_s2_val_for_me = vals_s2[best_opp_idx].item()
                best_s2_wp_for_me = win_probs_s2[best_opp_idx].item()
                
                expected_opp_win += best_s2_val_for_me * prob
                expected_opp_win_prob += best_s2_wp_for_me * prob
                
            # We want to MAXIMIZE the expected equity
            if expected_opp_win > best_expectation:
                best_expectation = expected_opp_win
                best_expectation_wp = expected_opp_win_prob
                best_idx_global = original_idx
                
        self.last_value = best_expectation
        self.last_win_prob = best_expectation_wp
        return best_idx_global

    def _run_3ply_beam(self, game, moves, style="aggressive"):
        """
        3-Ply Search with aggressive Beam Pruning.
        Depth: My Move -> Opponent Response -> My Response -> Eval.
        """
        # 1. Pruning (Same as 2-ply but tighter beam)
        # Run 1-Ply eval
        boards_1ply = []
        for seq in moves:
            boards_1ply.append(game.get_afterstate(seq))
        opponent_1ply = 1 - game.turn
        values_1ply, _ = self._evaluate_states(boards_1ply, opponent_1ply, opponent_1ply, style, current_score=game.score)
        scored_moves = []
        for i, val in enumerate(values_1ply.cpu().numpy()):
            scored_moves.append((val, i, moves[i]))
        scored_moves.sort(key=lambda x: x[0])
        
        # BEAM WIDTH: 2 (Very Selective)
        candidates = scored_moves[:2] 
        
        best_expectation = -float('inf')
        best_idx_global = candidates[0][1]
        
        sim_game = BackgammonGame()
        current_turn = game.turn
        opponent = 1 - current_turn
        
        # For each candidate (Ply 1)
        for (val_1ply, original_idx, seq) in candidates:
            b1, ba1, o1 = game.get_afterstate(seq)
            expected_val = 0.0
            
            # Opponent Rolls (Ply 2 Chance)
            for roll, prob in self.dice_dist.items():
                sim_game.board = b1.copy()
                sim_game.bar = ba1.copy()
                sim_game.off = o1.copy()
                sim_game.turn = opponent
                sim_game.score = game.score
                opp_moves = sim_game.get_legal_moves(roll)
                
                if not opp_moves:
                    # Pass -> My Turn, but Dice not rolled.
                    # Estimate value of state "It is My Turn"
                    val = self._evaluate_single(b1, ba1, o1, current_turn, style, current_score=game.score)
                     # Note: _evaluate_single returns value from PERSPECTIVE of `p`.
                     # Here p=current_turn (ME). So higher is better.
                    expected_val += val * prob # Add My Equity
                    continue

                # Find Opponent Best Response (Ply 2 Min)
                # Greedy 1-ply search for opponent
                s2_boards = []
                for om in opp_moves:
                    s2_boards.append(sim_game.get_afterstate(om))
                
                # Evaluate resulting states (which are "My Turn, No Dice")
                # We assume Opponent wants to MINIMIZE my value of resulting state.
                vals_s2, _ = self._evaluate_states(s2_boards, current_turn, current_turn, style, current_score=game.score)
                
                # Select Best Opponent Move (Lowest Value for Me)
                best_opp_idx = torch.argmin(vals_s2).item()
                best_s2_state = s2_boards[best_opp_idx] # (b, ba, o)
                
                # --- PLY 3: My Response (Max) ---
                # We are at state `best_s2_state` (My Turn).
                # To clear horizon effect, we simulate MY roll.
                
                expected_ply3_val = 0.0
                
                # Nested Loop: My Rolls (Ply 3 Chance)
                # Optimization: We can't do full 21 rolls here (Too slow: 2*21*21 = 882 evaluations).
                # 882 evals takes <1 sec if batched 2-ply style, but here we have overhead.
                # Actually 882 * batch_size.
                # If we rely on _evaluate_single for leaf, it's ok.
                # But to find *best* move, we must search legal moves.
                # 882 * 20 moves = 17,000 checks. Too slow.
                
                # OPTIMIZATION: Use the `vals_s2` (Static Eval of My Turn) we just computed?
                # If we do that, it's just 2-Ply! 
                # User wants 3-Ply.
                # To make 3-Ply efficient, we trust the `vals_s2`? No.
                
                # Compromise: We only expand "My Turn" for the single best Opponent Move?
                # Yes, we already identified `best_s2_state`.
                # We only expand that ONE state.
                
                sim_game_3 = BackgammonGame()
                sim_game_3.turn = current_turn
                
                for roll3, prob3 in self.dice_dist.items():
                    sim_game_3.board = best_s2_state[0].copy()
                    sim_game_3.bar = best_s2_state[1].copy()
                    sim_game_3.off = best_s2_state[2].copy()
                    sim_game_3.score = game.score
                    
                    my_moves = sim_game_3.get_legal_moves(roll3)
                    
                    if not my_moves:
                        # Pass -> Opponent Turn
                        val = self._evaluate_single(best_s2_state[0], best_s2_state[1], best_s2_state[2], opponent, style, current_score=game.score)
                        # Value for Opponent. My Value = -Value?
                        # Wait, Model output is "Prob Perspective Player Wins".
                        # If I pass, it is Opponent Turn.
                        # I want My Winning Prob. = 1.0 - OppWinningProb.
                        # _evaluate_single(..., p=opponent) returns Opponent Win Prob.
                        my_prob = 1.0 * (0.0 - val) # No, value is logit-sum? No, existing code logic is complex.
                        # Let's check _evaluate_states:
                        # returns "values". values = Sum(Probs * Weights).
                        # Style "aggressive": weights = [-3, -2, -1, 1, 2, 3].
                        # So Value > 0 is good for Perspective Player.
                        # If I Eval from Opponent Perspective, Value is Good for Opponent.
                        # My Value = -1 * OppValue.
                        expected_ply3_val += (-1.0 * val) * prob3
                        continue
                        
                    # Find My Best Move (Ply 3 Max)
                    # Use 0-Ply Eval on all leaf states
                    s3_boards = []
                    for mm in my_moves:
                        s3_boards.append(sim_game_3.get_afterstate(mm))
                        
                    # Eval from My Perspective (ME)
                    vals_s3, _ = self._evaluate_states(s3_boards, opponent, current_turn, style, current_score=game.score) 
                    # Note: Next turn is Opponent. So `player` arg to _evaluate_states could be `opponent`?
                    # `evaluate_states` signature: (boards, player, perspective, ...)
                    # `p`(player) is used for `get_obs`. Obs: "Whose turn is it?"
                    # After I move, it is Opponent's turn. So p=opponent.
                    # But I want value for ME (perspective=current_turn).
                    
                    best_my_val = torch.max(vals_s3).item()
                    expected_ply3_val += best_my_val * prob3
                    
                # End My Roll Loop
                expected_val += expected_ply3_val * prob # Add to total expectation
                
            if expected_val > best_expectation:
                best_expectation = expected_val
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

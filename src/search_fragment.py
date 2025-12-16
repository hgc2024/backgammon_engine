
    def get_2ply_candidates(self, game, moves=None, style="aggressive"):
        """
        Public method to get ranked candidates with full stats (Equity, Win%).
        Used by Gen6 Agentic Workflow.
        """
        if moves is None:
            moves = game.legal_moves
            if not moves:
                return []
                
        # --- 1-PLY PRUNING ---
        boards_1ply = []
        for seq in moves:
            boards_1ply.append(game.get_afterstate(seq))
            
        opponent_1ply = 1 - game.turn
        values_1ply, _ = self._evaluate_states(boards_1ply, opponent_1ply, opponent_1ply, style, current_score=game.score)
        
        scored_moves = []
        for i, val in enumerate(values_1ply.cpu().numpy()):
            scored_moves.append((val, i, moves[i]))
            
        # Minimize Opponent Equity
        scored_moves.sort(key=lambda x: x[0])
        
        TOP_K = 5
        candidates = scored_moves[:TOP_K]
        
        # --- 2-PLY FULL EVAL ---
        results = []
        sim_game = BackgammonGame()
        current_turn = game.turn
        opponent = 1 - current_turn
        
        for (val_1ply, original_idx, seq) in candidates:
            b1, ba1, o1 = game.get_afterstate(seq)
            expected_equity = 0.0
            expected_win_prob = 0.0
            
            for roll, prob in self.dice_dist.items():
                sim_game.board = b1.copy()
                sim_game.bar = ba1.copy()
                sim_game.off = o1.copy()
                sim_game.turn = opponent
                sim_game.score = game.score
                
                opp_moves = sim_game.get_legal_moves(roll)
                
                if not opp_moves:
                    v, wp = self._evaluate_states([(b1, ba1, o1)], opponent, opponent, style, current_score=game.score)
                    expected_equity += v.item() * prob
                    expected_win_prob += wp.item() * prob
                    continue
                
                s2_boards = []
                for om in opp_moves:
                    s2_boards.append(sim_game.get_afterstate(om))
                    
                vals_s2, win_probs_s2 = self._evaluate_states(s2_boards, current_turn, current_turn, style, current_score=game.score)
                
                best_opp_idx = torch.argmin(vals_s2).item()
                best_val = vals_s2[best_opp_idx].item()
                best_wp = win_probs_s2[best_opp_idx].item()
                
                expected_equity += best_val * prob
                expected_win_prob += best_wp * prob
                
            results.append({
                "index": original_idx,
                "move": seq,
                "equity": expected_equity,
                "win_prob": expected_win_prob
            })
            
        # Sort by Equity Descending (Max my equity)
        results.sort(key=lambda x: x["equity"], reverse=True)
        return results

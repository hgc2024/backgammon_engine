
import ollama
import numpy as np
from src.search import ExpectiminimaxAgent

class Gen6Agent:
    def __init__(self, engine: ExpectiminimaxAgent, model_name="llama3.2"):
        self.engine = engine
        self.model_name = model_name
        self.last_reasoning = ""
        self.dilemma_threshold = 0.03 # Equity diff threshold to trigger LLM

    @property
    def device(self):
        return self.engine.device

    @property
    def last_value(self):
        return self.engine.last_value

    @property
    def last_win_prob(self):
        return getattr(self.engine, "last_win_prob", 0.0)

    def get_state_value(self, game, style="aggressive", depth=2):
        return self.engine.get_state_value(game, style=style, depth=depth)

    def is_contact(self, game):
        """
        Returns True if there is still contact (checkers can hit or block).
        Returns False if it is a pure race (checkers have passed each other).
        
        Logic:
        P0 moves HIGH (23) -> LOW (0). P0 Checkers are at indices P0_idxs.
        P1 moves LOW (0) -> HIGH (23). P1 Checkers are at indices P1_idxs.
        
        Contact exists if max(P0_idxs) >= min(P1_idxs).
        Actually, P0 starts at 23, 13, 8, 6.
        P1 starts at 0, 11, 16, 18.
        They cross.
        Race condition: All P0 checkers are LOWER than all P1 checkers.
        i.e. max(P0) < min(P1).
        """
        p0_idxs = [i for i, c in enumerate(game.board) if c > 0]
        p1_idxs = [i for i, c in enumerate(game.board) if c < 0]
        
        # Add Bar checkers (Bar is "before" start)
        # P0 Bar: effectively index 24 (needs to enter 23..18) -> Wait.
        # P0 moves 23->0. Bar enters at 23 (24).
        # P1 Bar: effectively index -1 (needs to enter 0..5).
        
        if game.bar[0] > 0: p0_idxs.append(24)
        if game.bar[1] > 0: p1_idxs.append(-1)
        
        if not p0_idxs or not p1_idxs:
            return False # One side bear off completely?
            
        return max(p0_idxs) >= min(p1_idxs)

    def check_hit(self, game, move_seq):
        """
        Simulates the move sequence to see if it results in an opponent checker hitting the bar.
        """
        # We need a scratch version of the board
        # _apply_move_simulation returns (board, bar)
        b = game.board.copy()
        ba = game.bar.copy()
        turn = game.turn
        opp = 1 - turn
        
        original_opp_bar = ba[opp]
        
        for step in move_seq:
            # step is (start, end)
            # We call the instance method on 'game'.
            # _apply_move_simulation(self, board, bar, move) returns (new_board, new_bar)
            b, ba = game._apply_move_simulation(b, ba, step)
            
        return ba[opp] > original_opp_bar

    def analyze_board_features(self, game):
        """
        Analyzes board for strategic features:
        - Primes (Consecutive blocks)
        - Anchors (Blocks in opponent home)
        - Blots (Vulnerable single checkers)
        """
        features = []
        board = game.board
        turn = game.turn
        
        # 1. Primes
        # Scan for sequences of 2+ checkers
        # We look for "made points" (count >= 2) for the current player
        # A Prime is usually 3+ consecutive points.
        consecutive = 0
        prime_start = -1
        
        # Iterate indices based on player direction?
        # Visual board is 0..23.
        # P0 owns +ve. P1 owns -ve.
        
        made_points = []
        for i in range(24):
            cnt = board[i]
            if turn == 0 and cnt >= 2: made_points.append(i)
            elif turn == 1 and cnt <= -2: made_points.append(i)
        
        # Find consecutive
        # Sort made_points (already sorted by range)
        current_run = []
        primes = []
        for p in made_points:
            if not current_run:
                current_run = [p]
            else:
                if p == current_run[-1] + 1:
                    current_run.append(p)
                else:
                    if len(current_run) >= 3:
                        primes.append(current_run)
                    current_run = [p]
        if len(current_run) >= 3:
             primes.append(current_run)
             
        for pr in primes:
            features.append(f"- Strong Prime detected (Length {len(pr)}) from Point {pr[0]+1} to {pr[-1]+1}.")
            
        # 2. Anchors
        # Anchor: Made point in opponent's home board or outer board (defensive).
        # P0 Home: 0-5. P1 Home: 18-23.
        # If Turn=0 (P0), Anchor is in P1 home (18-23).
        # If Turn=1 (P1), Anchor is in P0 home (0-5).
        
        anchors = []
        if turn == 0:
            # Check 18-23
            for i in range(18, 24):
                if board[i] >= 2: anchors.append(i)
        else:
            # Check 0-5
            for i in range(0, 6):
                if board[i] <= -2: anchors.append(i)
                
        if anchors:
            pts = ", ".join([str(x+1) for x in anchors])
            features.append(f"- Defensive Anchor(s) held at Point(s) {pts} (Opponent Home Board).")
            
        # 3. Blots (Weaknesses)
        # Single checker.
        blots = []
        for i in range(24):
            cnt = board[i]
            if turn == 0 and cnt == 1: blots.append(i)
            elif turn == 1 and cnt == -1: blots.append(i)
            
        if blots:
             pts = ", ".join([str(x+1) for x in blots])
             features.append(f"- Vulnerable Blots (single checkers) at Point(s) {pts}.")
             
        # 4. Traps (Opponent Primes?)
        # Let's check if opponent has a prime in front of us.
        # Too complex for quick rule, implied by board state.
        
        if not features:
            return "No salient features (Primes/Anchors) detected."
            
        return "\n".join(features)

    def generate_board_desc(self, game):
        """
        Generates a detailed natural language description of the board.
        Includes Pip counts, Bar/Off counts, and explicit point listings.
        """
        turn = game.turn # 0 or 1
        me_color = "White (Pos)" if turn == 0 else "Red (Neg)"
        opp_color = "Red (Neg)" if turn == 0 else "White (Pos)"
        
        # 1. Score & Context
        # Note: No Doubling Cube in this mode.
        p_score = game.score[turn]
        o_score = game.score[1-turn]
        
        ctx = f"Match Context:\n- You are {me_color}.\n- Opponent is {opp_color}.\n"
        ctx += f"- Current Score: You {p_score} vs Opponent {o_score} (Cumulative Series).\n"
        ctx += "- Doubling Cube: DISABLED. Play for single wins and gammons only.\n"
        
        # 2. Pip Counts
        p0_pip, p1_pip = game.get_pip_counts()
        me_pip = p0_pip if turn == 0 else p1_pip
        opp_pip = p1_pip if turn == 0 else p0_pip
        
        ctx += f"- Pip Count: You {me_pip} (Lower is better) vs Opponent {opp_pip}.\n"
        ctx += f"  (You are {'ahead' if me_pip < opp_pip else 'behind'} in the race by {abs(me_pip - opp_pip)} pips).\n"
        
        # 3. Bar & Off
        me_bar = game.bar[turn]
        opp_bar = game.bar[1-turn]
        me_off = game.off[turn]
        opp_off = game.off[1-turn]
        
        ctx += f"- Checkers on Bar: You {me_bar}, Opponent {opp_bar}.\n"
        ctx += f"- Checkers Borne Off: You {me_off}, Opponent {opp_off}.\n\n"
        
        # 4. Strategic Features
        feats = self.analyze_board_features(game)
        ctx += "Key Strategic Features:\n" + feats + "\n\n"
        
        # 5. Detailed Board Position
        ctx += "Full Board Position (Standard Index 0-23):\n"
        points_desc = []
        for i in range(24):
            cnt = game.board[i]
            if cnt == 0: continue
            
            # Identify owner
            owner = 0 if cnt > 0 else 1
            count = abs(cnt)
            owner_str = "You" if owner == turn else "Opponent"
            
            # Point Number relative to standard board
            # Let's just use raw index for clarity, or standard notation?
            # Standard notation depends on perspective.
            # Let's use Raw Index (0-23) but explain: 
            # "Point 0 (White Home)" ... "Point 23 (White Outer)"
            pt_label = f"Point {i+1}"
            points_desc.append(f"  - {pt_label}: {count} {owner_str} checkers")
            
        ctx += "\n".join(points_desc)
        
        return ctx

    def get_action(self, game, depth=2, style="aggressive"):
        """
        Orchestrates the decision.
        """
        self.last_reasoning = ""
        
        # 0. Check for forced moves or trivial cases
        if not game.legal_moves:
            return None
            
        # 1. Get Tactical Candidates (Gen5 2-Ply)
        candidates = self.engine.get_2ply_candidates(game, style=style)
        
        if not candidates:
            return 0 # Should not happen if legal_moves exist
            
        # Deduplicate Transpositional Moves (Mirrors)
        # e.g., ((0,1), (1,2)) is effectively same as ((1,2), (0,1))
        # We assume standard moves are commutative in effect.
        unique_candidates = []
        seen_moves = set()
        
        # Helper to safely sort mixed int/str (bar/off)
        def move_sort_key(m):
            # m is (start, end)
            s, e = m
            # Map 'bar' -> 100, 'off' -> 200 (arbitrary large ints for sorting)
            s_val = 100 if s == 'bar' else s
            e_val = 200 if e == 'off' else e
            return (s_val, e_val)
        
        for cand in candidates:
            # Sort the atomic moves within the sequence to create a canonical signature
            # cand['move'] is list of (start, end).
            # We use a custom key because 'off' (str) cannot be compared to ints in Python 3.
            move_sig = tuple(sorted(cand['move'], key=move_sort_key))
            
            if move_sig not in seen_moves:
                seen_moves.add(move_sig)
                unique_candidates.append(cand)
                
        candidates = unique_candidates
        
        best = candidates[0]
        
        # If only 1 move after deduplication, return it
        if len(candidates) == 1:
            self.last_reasoning = "Only one distinct legal move."
            return best['index']
            
        second = candidates[1]
        
        # 2. Race Condition Check
        # If pure race, use Gen5 (Calculated Heuristic is superior to LLM).
        if not self.is_contact(game):
             self.last_reasoning = "Pure Race detected. Using Gen5 Heuristic (Optimal)."
             return best['index']
        
        # 3. Dilemma & Hit Detection
        equity_diff = best['equity'] - second['equity']
        
        # Detect Hits
        # Check if the Best Move hits
        best_hits = self.check_hit(game, best['move'])
        
        # Check if alternative (Option 2 or 3) hits
        alt_hits = False
        hit_candidate_idx = -1
        
        for i, cand in enumerate(candidates[1:3]): # Check 2nd and 3rd
            if self.check_hit(game, cand['move']):
                alt_hits = True
                hit_candidate_idx = i + 1 # 0-based offset from candidates[1:]? No. i=0 is second.
                # candidates[1] is Option 2.
                break
        
        # Trigger Conditions:
        # A. Equity is close (Standard Dilemma)
        # B. Best move MISSES, but an alternative HITS (Aggressive safeguard)
        
        trigger_reason = ""
        if equity_diff < self.dilemma_threshold:
            trigger_reason = f"Equity Gap ({equity_diff:.3f}) < Threshold."
        elif not best_hits and alt_hits:
            trigger_reason = "Alternative move offers a HIT while best move does not."
            
        if not trigger_reason:
            # Fast Path
            self.last_reasoning = f"Standard Gen5 Move. {('Best move hits.' if best_hits else 'No better hit found.')} Equity Delta {equity_diff:.3f}."
            return best['index']
            
        # 4. The Council Convenes (Slow Path)
        # Construct Prompt
        board_desc = self.generate_board_desc(game)
        
        c_text = "CANDIDATE MOVES (Analyzed by Gen5 Engine):\n"
        for i, c in enumerate(candidates[:3]): # Top 3 only
            is_hit = self.check_hit(game, c['move'])
            hit_str = " [HITS OPPONENT]" if is_hit else ""
            c_text += f"Option {i+1}: Sequence {c['move']}{hit_str} | Equity: {c['equity']:.3f} | Win Prob: {c['win_prob']*100:.1f}%\n"
            
        prompt = f"""
You are a Backgammon Grandmaster.
Analyze the following position and choose the best move from the candidates provided.

{board_desc}

{c_text}

CONTEXT:
- The Council was summoned because: {trigger_reason}
- There is NO DOUBLING CUBE. This is a match played for total points.
- Your goal is to maximize your long-term winning chances.
- Pay attention to "Key Strategic Features" (Primes, Anchors, Blots) listed above. Avoid getting trapped behind primes if behind in race.
- VALUE OF HITTING: If a move hits an opponent (sends them to the bar), consider if the tempo gain outweighs the risk. Hitting is often correct if it disrupts the opponent or escapes a checker.

TASK:
1. LIST the Candidate Moves provided above.
2. Briefly analyze the strategic theme.
3. Compare the pros/cons of Option 1 vs Option 2.
4. Select the best move number.
5. Output your decision in this format:

FINAL_MOVE: [Number]
REASONING: [Your explanation]
"""
        
        try:
            print("Gen6: Convening Council (Calling Ollama)...")
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt},
            ])
            
            content = response['message']['content']
            
            # Prepend context to reasoning so user sees it
            display_reasoning = f"--- COUNCIL SESSION ---\n{c_text}\n--- MISTRAL OPINION ---\n{content}"
            self.last_reasoning = display_reasoning
            
            # Parse Move
            import re
            match = re.search(r'FINAL_MOVE:\s*\[?(\d)\]?', content)
            if match:
                choice_idx = int(match.group(1)) - 1 # 1-based to 0-based
                if 0 <= choice_idx < len(candidates):
                    # Valid ID
                    rec_idx = candidates[choice_idx]['index']
                    print(f"Gen6: Council overruled/confirmed. Chose Candidate {choice_idx+1}.")
                    return rec_idx
            
            # Fallback if parsing fails
            print("Gen6: Could not parse LLM decision. Defaulting to Gen5 #1.")
            return best['index']
            
        except Exception as e:
            print(f"Gen6 Error: {e}")
            self.last_reasoning = f"Council unavailable ({str(e)}). Using Gen5 default."
            return best['index']

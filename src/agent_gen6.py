
import ollama
import numpy as np
from src.search import ExpectiminimaxAgent

class Gen6Agent:
    def __init__(self, engine: ExpectiminimaxAgent, model_name="mistral"):
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

    def board_to_text(self, game):
        """Converts board state to natural language description."""
        p = "You (White/Pos)" if game.turn == 0 else "You (Red/Neg)"
        opp = "Opponent (Red/Neg)" if game.turn == 0 else "Opponent (White/Pos)"
        
        # Board Pips
        match_score = f"Match Score: You {game.score[game.turn]} - Opponent {game.score[1-game.turn]}"
        
        # Describe Checkers
        # Simplified for now: just raw board + bar + off in English
        # Smart: "You have 2 checkers on the 24 point."
        
        desc = f"You are Player {game.turn}. {match_score}.\n"
        if game.bar[game.turn] > 0:
            desc += f"You have {game.bar[game.turn]} checkers on the BAR (Must enter).\n"
        
        # Points description (Relative to player Home Board)
        # Player 0 Home: 1-6. Outer: 7-12...
        # Let's describe crucial features: Primes, Blots, Anchors.
        # This is complex to do robustly in short code.
        # Fallback: Provide ASCII diagram!
        
        diagram = game.render_ascii()
        desc += "Board State:\n" + diagram
        return desc
        
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
            
        best = candidates[0]
        
        # If only 1 move, return it
        if len(candidates) == 1:
            self.last_reasoning = "Only one legal move."
            return best['index']
            
        second = candidates[1]
        
        # 2. Dilemma Check
        equity_diff = best['equity'] - second['equity']
        
        is_dilemma = equity_diff < self.dilemma_threshold
        
        if not is_dilemma:
            # Fast Path
            self.last_reasoning = f"Standard Gen5 Move. Equity Delta ({equity_diff:.3f}) > Threshold."
            return best['index']
            
        # 3. The Council Convenes (Slow Path)
        # Construct Prompt
        board_desc = self.board_to_text(game)
        
        c_text = "Candidate Moves (analyzed by Computer Engine):\n"
        for i, c in enumerate(candidates[:3]): # Top 3 only
            c_text += f"{i+1}. Move sequence: {c['move']}. Equity: {c['equity']:.3f}. Win Prob: {c['win_prob']*100:.1f}%.\n"
            
        prompt = f"""
You are a Backgammon Grandmaster.
Analyze the following position and choose the best move from the candidates provided.

{board_desc}

{c_text}

The computer evaluation considers these moves very close (Equity difference of {equity_diff:.3f}).
Your goal is to use strategic reasoning (safety vs aggression, structure, gammon risk, match score) to break the tie.

Task:
1. Briefly analyze the strategic theme (Prime, Race, Blitz, Holding Game).
2. Compare Move 1 and Move 2 (and 3 if relevant).
3. Select the best move number (1, 2, or 3).
4. Output your decision in this format:
FINAL_MOVE: [Number]
REASONING: [Your explanation]
"""
        
        try:
            print("Gen6: Convening Council (Calling Ollama)...")
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt},
            ])
            
            content = response['message']['content']
            self.last_reasoning = content
            
            # Parse Move
            # Look for FINAL_MOVE: [N]
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

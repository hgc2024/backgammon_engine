import trueskill
import json
import os
import torch
import random
from src.game import BackgammonGame, GamePhase
from src.search import ExpectiminimaxAgent
from src.evaluate import RandomAgent
from src.model import BackgammonValueNet

class TrueSkillArena:
    def __init__(self, leaderboard_file="leaderboard.json"):
        self.leaderboard_file = leaderboard_file
        self.env = trueskill.TrueSkill(draw_probability=0.0) # Backgammon rarely draws (only if agreed or hit limit)
        self.ratings = self._load_ratings()
        self.history = []

    def _load_ratings(self):
        if os.path.exists(self.leaderboard_file):
            try:
                with open(self.leaderboard_file, "r") as f:
                    data = json.load(f)
                    # Convert dict back to Rating objects
                    ratings = {}
                    for name, stats in data.items():
                        ratings[name] = self.env.create_rating(mu=stats["mu"], sigma=stats["sigma"])
                    return ratings
            except:
                return {}
        return {}

    def save_ratings(self):
        data = {}
        for name, rating in self.ratings.items():
            data[name] = {"mu": rating.mu, "sigma": rating.sigma}
        with open(self.leaderboard_file, "w") as f:
            json.dump(data, f, indent=4)

    def get_rating(self, agent_name):
        if agent_name not in self.ratings:
            self.ratings[agent_name] = self.env.create_rating()
        return self.ratings[agent_name]

    def register_agent(self, name):
        """Ensures agent is in leaderboard."""
        self.get_rating(name)

    def play_match(self, agent1, name1, agent2, name2, match_target=5):
        """
        Plays a match (First to X points) and updates ratings.
        """
        print(f"Arena Match: {name1} vs {name2} (Target {match_target})")
        
        # Load Ratings
        r1 = self.get_rating(name1)
        r2 = self.get_rating(name2)
        
        game = BackgammonGame(match_target=match_target)
        
        # Determine winner
        # We need a game loop similar to evaluate_vs_random
        # But generic for two callable agents.
        
        winner_name = None
        
        # Who is Player 0? Agent 1.
        # Who is Player 1? Agent 2.
        
        game_over = False
        while not game_over:
            # Handle Phases
            if game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
               # Simplified: Always Roll for now unless Agents support Cube
               game.step(0)
            elif game.phase == GamePhase.RESPOND_TO_DOUBLE:
               game.step(0) # Always Take
            elif game.phase == GamePhase.DECIDE_MOVE:
                moves = game.legal_moves
                if not moves:
                    game.turn = 1 - game.turn
                    game.phase = GamePhase.DECIDE_CUBE_OR_ROLL
                    continue
                
                # Turn
                act = None
                if game.turn == 0:
                    act = self._get_agent_action(agent1, game)
                else:
                    act = self._get_agent_action(agent2, game)
                    
                game.step(act)
                
            elif game.phase == GamePhase.GAME_OVER:
                # Who won?
                if game.score[0] > game.score[1]:
                    # Player 0 (Agent 1) won match?
                    # Wait, BackgammonGame updates score but resets game if not match end.
                    # We need to check if match target reached.
                    pass
                else:
                    pass
                    
                # BackgammonGame Logic:
                # If score >= target, done?
                if max(game.score) >= match_target:
                    game_over = True
                    if game.score[0] > game.score[1]:
                        winner_name = name1
                    else:
                        winner_name = name2
                else:
                    # Next game in match
                    game.reset_game()
                    
        print(f"Winner: {winner_name}")
        
        # Update TrueSkill
        if winner_name == name1:
            new_r1, new_r2 = trueskill.rate_1vs1(r1, r2)
        else:
            new_r2, new_r1 = trueskill.rate_1vs1(r2, r1)
            
        self.ratings[name1] = new_r1
        self.ratings[name2] = new_r2
        self.save_ratings()
        
        print(f"New Ratings: {name1}: {new_r1.mu:.1f}, {name2}: {new_r2.mu:.1f}")

    def _get_agent_action(self, agent, game):
        # Wrapper to handle different agent signatures
        # Expectiminimax: get_action(game, depth=2)
        # Random: Just needs legal moves.
        # Heuristic: needs game.
        
        # Check type
        if hasattr(agent, 'get_action'):
            # Our Search Agent
            return agent.get_action(game, depth=1) # Use 1-Ply for speed in Arena for now
        elif hasattr(agent, 'predict'):
             # SB3 PPO?
             pass
        else:
             # Random/Simple Function?
             if hasattr(agent, '__call__'):
                 return agent(game)
                 
        # Fallback Random
        return random.randint(0, len(game.legal_moves)-1)

def run_tournament():
    arena = TrueSkillArena()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Random Agent
    random_agent = RandomAgent()
    arena.register_agent("Random")
    
    # 2. Current TD Model (1-Ply Greedy)
    # We load "td_backgammon.pth"
    if os.path.exists("td_backgammon.pth"):
        td_agent = ExpectiminimaxAgent("td_backgammon.pth", device=device)
        arena.register_agent("TD-1Ply")
        
        # Play Match
        arena.play_match(td_agent, "TD-1Ply", random_agent, "Random", match_target=3)
        
    else:
        print("No TD model found.")

if __name__ == "__main__":
    run_tournament()

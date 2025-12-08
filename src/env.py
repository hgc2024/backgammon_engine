import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional

from .game import BackgammonGame, GamePhase
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env):
    return env.action_masks()

def create_env():
    env = BackgammonEnv()
    env = ActionMasker(env, mask_fn)
    return env

class BackgammonEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, match_target=15):
        super(BackgammonEnv, self).__init__()
        self.game = BackgammonGame(match_target=match_target)
        
        # Observation Space: 198 floats
        # 192 (Board) + 2 (Bar) + 2 (Off) + 1 (Turn) + 1 (Cube?) -> Let's check typical setup
        # TD-Gammon: 198.
        self.observation_space = spaces.Box(low=0, high=1, shape=(198,), dtype=np.float32)
        
        # Action Space: Discrete
        # Max Moves: For now set to 200 possible sequences? 
        # Actually in Double phase, action is 0 or 1.
        # In Move phase, action is index of move_sequence.
        # We need a unified action space size. 
        # Let's say max 100 moves.
        # If phase is CUBE, actions 0-1 are valid.
        # If phase is MOVE, actions 0..N are valid.
        self.action_space = spaces.Discrete(200) 
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        start = 0
        if options and 'starting_player' in options:
            start = options['starting_player']
        self.game.reset_match(starting_player=start)
        # If we want to start from a specific game in match? For now start fresh.
        # Actually, self.game.step() handles phase transitions.
        # Initial phase is DECIDE_CUBE_OR_ROLL.
        # But we need to handle "Agent self-play".
        # This Env returns observation relative to CURRENT player.
        return self._get_obs(), {}
        
    def step(self, action):
        """
        Action is an integer index.
        """
        # Validate action
        valid_actions_count = 0
        if self.game.phase in [GamePhase.DECIDE_CUBE_OR_ROLL, GamePhase.RESPOND_TO_DOUBLE]:
            valid_actions_count = 2 # 0 or 1
        elif self.game.phase == GamePhase.DECIDE_MOVE:
            valid_actions_count = len(self.game.legal_moves)
            if valid_actions_count == 0:
                # Should have been handled by game auto-passing?
                # The game logic handles "no legal moves" by switching turn automatically inside _roll_and_start_turn
                # BUT if we are in DECIDE_MOVE, it means legal_moves is not empty.
                pass
        
        # Safe-guard for invalid action
        if action >= valid_actions_count:
            # Invalid action punishment?
            # With MaskablePPO, this shouldn't happen if masks are correct.
            # If it happens, treat as 0 or random?
            action = 0 
            
        reward, winner, done = self.game.step(action)
        
        # Reward shaping?
        # If self-play, reward is only final.
        # For now, raw reward.
        
        obs = self._get_obs()
        return obs, float(reward), done, False, {}
        
    def action_masks(self):
        """
        Returns boolean mask of valid actions.
        Required for MaskablePPO.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        if self.game.phase in [GamePhase.DECIDE_CUBE_OR_ROLL, GamePhase.RESPOND_TO_DOUBLE]:
            # 0, 1 valid?
            # Check logic
            can_double = self.game._can_double(self.game.turn)
            
            if self.game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
                mask[0] = True # Roll always allowed
                if can_double:
                    mask[1] = True # Double
            else:
                # Respond
                mask[0] = True # Take
                mask[1] = True # Drop
                
        elif self.game.phase == GamePhase.DECIDE_MOVE:
            n_moves = len(self.game.legal_moves)
            if n_moves > 0:
                mask[:n_moves] = True
            else:
                # Should not happen if logic is correct
                pass
                
        return mask

    def _get_obs(self):
        """
        Constructs the 198-dimension feature vector.
        Perspectives:
        If turn=0 (White), board is 0-23. (+ is own, - is opp).
        If turn=1 (Black), board is 0-23. (- is own, + is opp).
        
        We need to normalize the view so "My 1 point" is always same index?
        Standard RL practice: "Egocentric view".
        
        TD-Gammon Features:
        - Player Own Pieces (0-23) -> 4 * 24 = 96 units
        - Opponent Pieces (0-23 mirrored) -> 4 * 24 = 96 units
        - Bar (Player) -> 1 unit? 2 units?
        - Bar (Opponent)
        - Off (Player)
        - Off (Opponent)
        - Turn info?
        
        Total 198.
        96 + 96 = 192.
        Bar (2) + Off (2) = 4.
        Turn (1)? Color (1)? 
        Let's assume:
        192 (Board) + 2 Bar + 2 Off + 2 (White/Black? or Turn?)
        
        Implementation:
        For each point 0..23 (from My perspective, moving 24->0 or 1->24?
        Let's stick to standard:
        Index 0 = My 24 point (farthest). Index 23 = My 1 point (closest).
        
        wait, My "Home" is usually 1-6 or 19-24.
        Game logic:
        P0 Home 0-5.
        P1 Home 18-23.
        
        Let's map everything to "Distance from off".
        P0: 0 is dist 0 (Home). 23 is dist 23.
        P1: 0 is dist 23. 23 is dist 0 (Home).
        
        So if Turn=0:
            MyPoints: board[0]..board[23] (reversed? No, 0 is home).
            OppPoints: board[0]..board[23].
            
            Board[0] = +2 means P0 has 2 on pt 0.
            
        If Turn=1:
            MyPoints: board[::-1] * -1?
            Board[23] = -2 means P1 has 2 on pt 23. (Which is P1's home).
            
            So we want input[0] to correspond to "My Point 1" (or 24?).
            Let's use "Point 1" = Furthest. "Point 24" = Home.
            
        Let's use `game` internal representation logic for consistency.
        
        """
        board = self.game.board
        bar = self.game.bar
        off = self.game.off
        turn = self.game.turn
        
        # Egocentric Board
        # We want an array `my_board` where `my_board[i]` is count at point i
        # Scheme: Points 0-23.
        # If Turn=0: 0-23 is naturally correct direction? (moving 23->0).
        # If Turn=1| 0-23 is Reverse. (moving 0->23).
        
        # Let's align to "Point 1 is Opponent Home, Point 24 is My Home".
        # P0: Moves 23->0. So 23 is "Start", 0 is "End".
        # P1: Moves 0->23. So 0 is "Start", 23 is "End".
        
        if turn == 0:
            # My pieces are Positive.
            my_board = np.maximum(board, 0)
            opp_board = np.abs(np.minimum(board, 0))
            
            # Map to 24 indices?
            # 23 (Start) -> 0 (End).
            # Let's assume input needs 24 points ordered.
            # Order: 23 down to 0? or 0 up to 23?
            # Let's pick 0..23.
            pass
        else:
            # My pieces are Negative.
            my_board = np.abs(np.minimum(board, 0))
            opp_board = np.maximum(board, 0)
            
            # P1 perspective: 0 is Start, 23 is End.
            # P0 perspective: 23 is Start, 0 is End.
            # To match, we reverse P1 views?
            # P0: 23(Start)..0(End)
            # P1: 0(Start)..23(End)
            # If we pass my_board directly:
            # P0: 23 has checker. P1: 0 has checker.
            # If we leave as is, P0's "Start" is index 23. P1's "Start" is index 0.
            # This is NOT symmetric. We MUST flip P1 so index 23 becomes its Start logic?
            # Or Flip P1 so P1[0] maps to physical 23?
            
            my_board = my_board[::-1] # Now index 0 is physical 23 (End). Index 23 is physical 0 (Start).
            opp_board = opp_board[::-1]
            
            # Wait, P0: Index 0 is End. Index 23 is Start.
            # P1 (flipped): Index 0 is "Physical 23" (End). Index 23 is "Physical 0" (Start).
            # So now they are aligned. Index 0 = End (Home). Index 23 = Start.
            
        features = []
        
        # Helper for 4 units
        def encode_point(count):
            # [1 if >=1, 1 if >=2, 1 if >=3, (count-3)/2 if >3 else 0]
            f = [0]*4
            if count >= 1: f[0] = 1
            if count >= 2: f[1] = 1
            if count >= 3: f[2] = 1
            if count > 3: f[3] = (count - 3) / 2.0
            return f
            
        # 1. My Board (24 * 4)
        for c in my_board:
            features.extend(encode_point(c))
            
        # 2. Opp Board (24 * 4)
        for c in opp_board:
            features.extend(encode_point(c))
            
        # 3. Bar
        # My Bar
        my_bar_cnt = bar[turn]
        # Opp Bar
        opp_bar_cnt = bar[1-turn]
        features.extend([my_bar_cnt / 2.0, opp_bar_cnt / 2.0])
        
        # 4. Off
        my_off_cnt = off[turn]
        opp_off_cnt = off[1-turn]
        features.extend([my_off_cnt / 15.0, opp_off_cnt / 15.0])
        
        # 5. Turn / Cube
        # Maybe "Am I on roll?" (Always 1 in this setup?)
        features.append(1.0) 
        # Cube Value?
        features.append(self.game.cube_value / 64.0) # Normalized
        
        return np.array(features, dtype=np.float32)
       
    def render(self, mode='human'):
        print(f"Turn: {self.game.turn} | Score: {self.game.score}")
        print(f"Board: {self.game.board}")
        print(f"Bar: {self.game.bar} | Off: {self.game.off}")

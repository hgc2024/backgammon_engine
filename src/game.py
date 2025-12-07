import numpy as np
import random
from typing import List, Tuple, Optional

import enum

class GamePhase(enum.Enum):
    DECIDE_CUBE_OR_ROLL = 0
    ROLL_DICE = 1 # Intermediate auto-state, usually skipped to MOVES
    DECIDE_MOVE = 2
    RESPOND_TO_DOUBLE = 3
    GAME_OVER = 4

class BackgammonGame:
    """
    Backgammon Game Engine with Doubling Cube and Crawford Rule support.
    
    Standard representation:
    - Board is 24 points.
    - Positive numbers = Player 0 (White) checkers.
    - Negative numbers = Player 1 (Black/Red) checkers.
    - Points 0-23.
    - Bar is stored separately.
    - Off is stored separately.
    """
    
    def __init__(self, match_target: int = 15):
        self.match_target = match_target
        self.score = [0, 0]  # [Player 0, Player 1]
        self.crawford_active = False # True if we are CURRENTLY in the Crawford game
        self.crawford_played = False # True if the Crawford game has already been played
        self.reset_match()

    def reset_match(self):
        self.score = [0, 0]
        self.crawford_active = False
        self.crawford_played = False
        self.reset_game()

    def reset_game(self):
        """Resets the board for a new game within the match."""
        # 24 points: 0-23
        self.board = np.zeros(24, dtype=int)
        
        # Initial Setup (Player 0 perspective - moving from 23 down to 0)
        # 2 at 23, 5 at 12, 3 at 7, 5 at 5
        self.board[23] = 2
        self.board[12] = 5
        self.board[7] = 3
        self.board[5] = 5
        
        # Opponent (Player 1 - moving from 0 up to 23)
        # Mirror: 2 at 0, 5 at 11, 3 at 16, 5 at 18
        self.board[0] = -2
        self.board[11] = -5
        self.board[16] = -3
        self.board[18] = -5

        self.bar = [0, 0] # [Player 0 count, Player 1 count]
        self.off = [0, 0] # [Player 0 count, Player 1 count]
        
        self.turn = 0 # 0 or 1
        self.cube_value = 1
        self.cube_owner = -1 # -1: centered, 0: player 0, 1: player 1
        
        # Crawford Rule Check
        # If a player is 1 point away from winning, the NEXT game is the Crawford game.
        leader_score = max(self.score)
        trailer_score = min(self.score)
        points_to_win = self.match_target - leader_score
        
        if points_to_win == 1 and not self.crawford_played:
            self.crawford_active = True
            self.crawford_played = True # Mark as played (active for this game)
        else:
            self.crawford_active = False
            
        self.phase = GamePhase.DECIDE_CUBE_OR_ROLL
        self.current_roll = []
        self.legal_moves = []

            
    def roll_dice(self) -> Tuple[int, int]:
        return (random.randint(1, 6), random.randint(1, 6))



    def double_cube(self, player: int):
        """
        Player attempts to double.
        Returns Success boolean.
        """
        if self.crawford_active:
            return False # No doubling in Crawford game
            
        if self.cube_owner != -1 and self.cube_owner != player:
            return False # Not your cube
            
        # Logic for offering double...
        # In a game loop, this would trigger a decision state for the opponent.
        return True

    def accept_double(self, player: int):
        """Opponent accepts double."""
        self.cube_value *= 2
        self.cube_owner = player # Player who accepted now owns it

    def step(self, action_idx: int) -> Tuple[int, int, bool]:
        """
        Executes a step in the game.
        Returns: (reward for current player, winner_if_any, done)
        """
        # Interpretation of action depends on phase
        
        if self.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
            # Actions: 0 = Roll, 1 = Double
            if action_idx == 1:
                # Double
                if not self._can_double(self.turn):
                    # Illegal double attempt -> Treat as Roll? Or error?
                    self._roll_and_start_turn()
                else:
                    self.phase = GamePhase.RESPOND_TO_DOUBLE
                    self.turn = 1 - self.turn # Switch control to opponent
            else:
                self._roll_and_start_turn()
                
        elif self.phase == GamePhase.RESPOND_TO_DOUBLE:
            # Actions: 0 = Generic "Take", 1 = Generic "Drop"
            if action_idx == 0:
                # Take
                self.cube_value *= 2
                self.cube_owner = self.turn
                self.turn = 1 - self.turn # Return control to doubler
                self._roll_and_start_turn()
            else:
                # Drop
                winner = 1 - self.turn
                reward_val = self.cube_value # No gammon on drop
                self.score[winner] += reward_val
                return reward_val, winner, True
                
        elif self.phase == GamePhase.DECIDE_MOVE:
            # Action matches index in self.legal_moves
            if not self.legal_moves:
                # No legal moves possible (blocked)
                self.turn = 1 - self.turn
                self.phase = GamePhase.DECIDE_CUBE_OR_ROLL
            else:
                move_seq = self.legal_moves[action_idx]
                for move in move_seq:
                    start, end = move
                    if end == 'off':
                        self.off[self.turn] += 1
                    self.board, self.bar = self._apply_move_simulation(self.board, self.bar, move)
                    
                # Check Win
                winner, pts = self.check_win()
                if winner != -1:
                    self.score[winner] += pts
                    self.phase = GamePhase.GAME_OVER
                    return pts, winner, True
                    
                # Switch Turn
                self.turn = 1 - self.turn
                self.phase = GamePhase.DECIDE_CUBE_OR_ROLL
                
        return 0, -1, False

    def _roll_and_start_turn(self):
        self.current_roll = self.roll_dice()
        self.legal_moves = self.get_legal_moves(self.current_roll)
        
        if not self.legal_moves:
             # Blocked. Turn passes.
             self.turn = 1 - self.turn
             self.phase = GamePhase.DECIDE_CUBE_OR_ROLL
        else:
             self.phase = GamePhase.DECIDE_MOVE

    def _can_double(self, player):
        if self.crawford_active: return False
        if self.cube_owner != -1 and self.cube_owner != player: return False
        return True

    def get_legal_moves(self, roll: Tuple[int, int]) -> List[List[Tuple[int, int]]]:
        """
        Returns all legal move sequences for the current turn.
        Enforces the "must use max dice" rule.
        Returns a list of move sequences. Each sequence is a list of (start, end).
        """
        dice = list(roll)
        if dice[0] == dice[1]:
            dice = [dice[0]] * 4
            
        possible_sequences = [] # List of (moves_list, resulting_board_state)
        
        # Recursive search for moves
        # We search depth-first.
        # State: (current_board, current_bar, remaining_dice, current_path)
        
        # Optimization: We only care about the sequence of moves, but the state matters for validation.
        # Since the board is small, we can probably get away with deepcopying or undoing moves.
        
        # However, for 4 dice, branching factor can be high.
        # Standard approach:
        # Try to use die[0]. If successful, recurse with remaining dice.
        
        direction = -1 if self.turn == 0 else 1
        home_range = range(0, 6) if self.turn == 0 else range(18, 24)
        bar_idx = 0 if self.turn == 0 else 1
        
        permutations = set()
        if len(dice) == 2 and dice[0] != dice[1]:
            permutations.add(tuple(dice))
            permutations.add((dice[1], dice[0]))
        else:
            permutations.add(tuple(dice))
            
        final_sequences = []
        max_moves_found = 0
        
        for p_dice in permutations:
            # We need to explore this dice order.
            # But wait, Backgammon rule: if you can play (d1, d2) OR (d2, d1), you can choose.
            # AND if you can only play one, you must play the larger one (if possible).
            # This logic is tricky.
            
            # Let's collect ALL reachable states and path lengths.
            found_paths = self._find_moves_recursive(self.board.copy(), self.bar.copy(), list(p_dice))
            
            for path in found_paths:
                if len(path) > max_moves_found:
                    max_moves_found = len(path)
                    final_sequences = [path]
                elif len(path) == max_moves_found:
                    final_sequences.append(path)
                    
        # Filter duplicates (sequences might differ but moves are same?) 
        # Actually (start, end) might be same but generated from different dice orders.
        # In BG, which die was used for which move usually doesn't matter for the final state,
        # but for RL training we might want to just output the moves.
        
        # Special Rule: If max_moves_found < len(dice)
        # Check if we could have played a larger die but didn't.
        # If we have dice [3, 6] and we played [3] but could have played [6], we must drop the [3] solution.
        # This is handled if we strictly enforce "find max depth" across all permutations.
        # BUT: If we have [3, 6] and can play 3->X or 6->Y (but not both), we MUST play 6.
        
        if len(dice) == 2 and dice[0] != dice[1] and max_moves_found == 1:
            # We played 1 move. Was it the larger die?
            # Filter final_sequences to ensure we used the larger die if possible.
            # This is complex to check post-hoc without tracking which die was used.
            # Let's assume _find_moves_recursive tracks used dice or implies it.
            
            # Re-run logic with explicit "must use max" check?
            # Simple heuristic: If we have multiple 1-move paths, and one used 6 and one used 3, keep 6.
            pass # TODO: Refine this. For now assume random/first valid.
            
            # To fix: check the move distance?
            # abs(start - end). If start is bar, distance is calculated from edge.
            
            larger_die = max(dice)
            smaller_die = min(dice)
            
            valid_larger = False
            for seq in final_sequences:
                move = seq[0]
                dist = self._get_move_distance(move)
                if dist == larger_die:
                    valid_larger = True
                    break
            
            if valid_larger:
                 final_sequences = [s for s in final_sequences if self._get_move_distance(s[0]) == larger_die]
                 
        unique_sequences = list(set(tuple(seq) for seq in final_sequences))
        # custom sort key: handles 'off' (str) vs int in tuples
        def sort_key(seq):
            # seq is tuple of moves ((start, end), ...)
            # convert 'off' to special int for sorting
            return tuple((s, e if isinstance(e, int) else -999) for s, e in seq)
            
        return sorted(unique_sequences, key=sort_key)

    def _get_move_distance(self, move):
        start, end = move
        # Handle Bar
        player = self.turn
        if start == 'bar':
            if player == 0:
                # 24 -> ?
                # If end is 'off' (impossible logic but safe to check)
                if end == 'off': return 25 
                return 24 - end
            else:
                # -1 -> ?
                if end == 'off': return 25
                return end - (-1)
        elif end == 'off':
            # Not exact distance if bearing off with larger die
            return 0 # Special case
        else:
            return abs(start - end)

    def _find_moves_recursive(self, board, bar, dice) -> List[List[Tuple[int, int]]]:
        if not dice:
            return [[]]
            
        die = dice[0]
        remaining_dice = dice[1:]
        
        moves = self._generate_single_moves(board, bar, die)
        
        if not moves:
            return [[]] # No moves for this die, stop here.
            
        paths = []
        for move in moves:
            # Apply move
            new_board, new_bar = self._apply_move_simulation(board, bar, move)
            
            # Recurse
            sub_paths = self._find_moves_recursive(new_board, new_bar, remaining_dice)
            
            for sub in sub_paths:
                paths.append([move] + sub)
                
        return paths

    def _generate_single_moves(self, board, bar, die):
        player = self.turn
        moves = []
        direction = -1 if player == 0 else 1
        
        # 1. Must enter from bar if checker on bar
        if bar[player] > 0:
            # Player 0 enters at 24-die. Player 1 enters at 0+die-1 = die-1.
            entry_point = (24 - die) if player == 0 else (die - 1)
            
            if self._is_valid_dest(board, entry_point):
                moves.append(('bar', entry_point))
            return moves # Must move from bar if possible
            
        # 2. Normal moves
        # Iterate over all points
        points = reversed(range(24)) if player == 0 else range(24)
        
        # Check bearing off eligibility
        can_bear_off = self._can_bear_off(board, bar, player)
        
        for i in points:
            if (player == 0 and board[i] > 0) or (player == 1 and board[i] < 0):
                # Try standard move
                dest = i + (die * direction)
                
                # Check Bounds
                if 0 <= dest < 24:
                    if self._is_valid_dest(board, dest):
                        moves.append((i, dest))
                elif can_bear_off:
                    # Bearing Off Logic
                    # Exact off?
                    exact = (dest == -1 and player == 0) or (dest == 24 and player == 1)
                    if exact:
                         moves.append((i, 'off'))
                    else:
                        # Over-shoot? Only allowed if no pieces on higher points.
                        # dest is numeric here?
                        # Player 0: dest < -1.
                        # Player 1: dest > 24.
                        
                        overshoot = False
                        if player == 0:
                            if dest < -1:
                                overshoot = True
                        else:
                            if dest > 24:
                                overshoot = True
                                
                        if overshoot:
                             # Check if any piece is "behind" this one (further from goal)
                             # Player 0: higher indices in home board.
                             # Player 1: lower indices in home board.
                             
                             is_furthest = True
                             if player == 0:
                                 for prev in range(i + 1, 6):
                                     if board[prev] > 0:
                                         is_furthest = False
                                         break
                             else:
                                 for prev in range(18, i):
                                     if board[prev] < 0:
                                         is_furthest = False
                                         break
                                         
                             if is_furthest:
                                 moves.append((i, 'off'))
                                 
        return moves

    def _is_valid_dest(self, board, p_idx):
        # Valid if empty, own color, or opponent has only 1 checker (hit)
        player = self.turn
        cnt = board[p_idx]
        if cnt == 0: return True
        if (player == 0 and cnt > 0) or (player == 1 and cnt < 0): return True
        if abs(cnt) == 1: return True # Hit
        return False # Blocked

    def _can_bear_off(self, board, bar, player):
        if bar[player] > 0: return False
        
        # Check if all checkers are in home board
        # Player 0: 0-5. All >5 must be 0.
        # Player 1: 18-23. All <18 must be 0.
        
        if player == 0:
            for i in range(6, 24):
                if board[i] > 0: return False
        else:
            for i in range(0, 18):
                if board[i] < 0: return False
        return True

    def _apply_move_simulation(self, board, bar, move):
        """Returns new board/bar copy after move."""
        b = board.copy()
        ba = bar.copy()
        player = self.turn
        
        start, end = move
        
        # Remove from start
        if start == 'bar':
            ba[player] -= 1
        else:
            if player == 0: b[start] -= 1
            else: b[start] += 1 # Negative
            
        # Add to end
        if end == 'off':
            pass # Removed from board, not added anywhere on board
        else:
            # Handle Hit
            if (player == 0 and b[end] == -1):
                b[end] = 0
                ba[1] += 1 # Hit opponent to bar
            elif (player == 1 and b[end] == 1):
                b[end] = 0
                ba[0] += 1
                
            if player == 0: b[end] += 1
            else: b[end] -= 1
            
        return b, ba

    def check_win(self):
        """Checks if current player has borne off all checkers."""
        if self.off[0] >= 15:
            return 0, self._calculate_score(0)
        if self.off[1] >= 15:
            return 1, self._calculate_score(1)
        return -1, 0

    def _calculate_score(self, winner: int):
        """
        Calculates score multiplier (1, 2 for gammon, 3 for backgammon) * cube.
        """
        loser = 1 - winner
        multiplier = 1
        
        if self.off[loser] == 0:
            multiplier = 2 # Gammon
            # Check Backgammon: Loser has checker on winner's home board or bar
            # Player 0 home: 0-5. Player 1 home: 18-23.
            
            checkers_on_bar = self.bar[loser] > 0
            checkers_in_winner_home = False
            
            if winner == 0:
                # Winner is White (0-23). Home is 0-5.
                # Loser is Black. Moving 0->23. Backgammon if Black has pieces in 0-5.
                if np.any(self.board[0:6] < 0): 
                    checkers_in_winner_home = True
            else:
                # Winner is Black. Home is 18-23.
                # Loser is White. Moving 23->0. Backgammon if White has pieces in 18-23.
                if np.any(self.board[18:24] > 0):
                    checkers_in_winner_home = True
                    
            if checkers_on_bar or checkers_in_winner_home:
                multiplier = 3 # Backgammon
                
        return self.cube_value * multiplier

    def render_ascii(self) -> str:
        """
        Returns an ASCII representation of the board.
        Points 0-23 (Standard annotation often 1-24).
        Let's display 13-24 on top (Left to Right? or Right to Left?)
        Standard BG:
        Top: 13 14 15 16 17 18 | 19 20 21 22 23 24
        Bot: 12 11 10 09 08 07 | 06 05 04 03 02 01
        
        Our internal: 0-23.
        Player 0 (Positive) moves 23->0.
        Player 1 (Negative) moves 0->23.
        
        Let's show indices 0-23 clearly.
        """
        lines = []
        lines.append(f"Score: You {self.score[0]} - {self.score[1]} Cpu | Cube: {self.cube_value} (Owner: {self.cube_owner})")
        lines.append(f"Turn: {'You (White/Pos)' if self.turn == 0 else 'Cpu (Black/Neg/Red)'}")
        lines.append("Bar: " + str(self.bar))
        lines.append("Off: " + str(self.off))
        lines.append("-" * 65)
        
        # Top half: 12..23 ( indices 12 to 23 )
        # Traditionally 13..24. Let's label them 12..23 for internal consistency debugging.
        
        top_indices = range(12, 24)
        top_str = " | ".join([f"{i:2}" for i in top_indices])
        lines.append("Idx: " + top_str)
        
        checkers_top = []
        for i in top_indices:
            val = self.board[i]
            sym = '.'
            if val > 0: sym = f"W{val}"
            elif val < 0: sym = f"B{abs(val)}"
            checkers_top.append(f"{sym:^2}")
        lines.append("Val: " + " | ".join(checkers_top))
        
        lines.append("-" * 65)
        
        # Bottom half: 11..0
        bot_indices = range(11, -1, -1)
        checkers_bot = []
        for i in bot_indices:
            val = self.board[i]
            sym = '.'
            if val > 0: sym = f"W{val}"
            elif val < 0: sym = f"B{abs(val)}"
            checkers_bot.append(f"{sym:^2}")
        lines.append("Val: " + " | ".join(checkers_bot))
        
        bot_str = " | ".join([f"{i:2}" for i in bot_indices])
        lines.append("Idx: " + bot_str)
        lines.append("-" * 65)
        
        return "\n".join(lines)


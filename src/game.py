import numpy as np
import random
import itertools
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

    def reset_match(self, starting_player: int = 0):
        self.score = [0, 0]
        self.crawford_active = False
        self.crawford_played = False
        self.reset_game(starting_player=starting_player)

    def reset_game(self, starting_player: int = 0):
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
        
        self.turn = starting_player # 0 or 1
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
        self.dice = [] # Current remaining dice (expanded e.g. [6,6,6,6])
        self.legal_moves = []
        self.winner = -1

            
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
            if action_idx not in (0, 1):
                raise ValueError("cube decision must be 0 (roll) or 1 (double)")
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
            if action_idx not in (0, 1):
                raise ValueError("double response must be 0 (take) or 1 (drop)")
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
                self.phase = GamePhase.GAME_OVER
                self.winner = winner
                return reward_val, winner, True
                
        elif self.phase == GamePhase.DECIDE_MOVE:
            # Action matches index in self.legal_moves
            if not self.legal_moves:
                # No legal moves possible (blocked)
                self.turn = 1 - self.turn
                self.phase = GamePhase.DECIDE_CUBE_OR_ROLL
            else:
                if not isinstance(action_idx, int) or not 0 <= action_idx < len(self.legal_moves):
                    raise ValueError(
                        f"move action index {action_idx!r} is outside "
                        f"0..{len(self.legal_moves) - 1}"
                    )
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
                    self.winner = winner
                    return pts, winner, True
                    
                # Switch Turn
                self.turn = 1 - self.turn
                self.phase = GamePhase.DECIDE_CUBE_OR_ROLL
                self.legal_moves = [] # Clear stale moves
                
        return 0, -1, False

    def get_legal_partial_moves(self) -> List[Tuple[int, int]]:
        """
        Return legal first atomic moves for the current remaining dice.

        A move is exposed only if it begins a complete continuation that obeys
        maximum-dice usage and the larger-die rule.
        """
        return list(self._get_legal_partial_options())

    @staticmethod
    def _move_sort_key(move):
        start, end = move
        start_key = -1 if start == "bar" else int(start)
        end_key = 25 if end == "off" else int(end)
        return start_key, end_key

    def _get_legal_partial_options(self):
        """Map each legal first move to dice that may legally perform it."""
        options = {}
        for path in self._get_legal_labeled_paths(
            self.board, self.bar, tuple(self.dice)
        ):
            if not path:
                continue
            move, die = path[0]
            options.setdefault(move, set()).add(die)
        return dict(sorted(options.items(), key=lambda item: self._move_sort_key(item[0])))

    def step_partial(self, move: Tuple[int, int]) -> Tuple[int, int, bool]:
        """
        Applies a SINGLE atomic move (Human Play).
        1. Validates move is legal for one of the active dice.
        2. Updates Board.
        3. Removes Used Die.
        4. Checks if turn is finished.
        """
        if self.phase != GamePhase.DECIDE_MOVE:
            raise ValueError("partial moves are only legal during DECIDE_MOVE")

        move = tuple(move)
        options = self._get_legal_partial_options()
        valid_dice = options.get(move)
        if not valid_dice:
            raise ValueError(
                f"illegal partial move {move!r} for dice {self.dice}; "
                f"legal moves are {list(options)}"
            )

        start, end = move
        distance = self._get_move_distance(move)
        die_used = (
            distance if distance in valid_dice else min(valid_dice)
        )
        self.dice.remove(die_used)
            
        # 2. Apply Move
        self.board, self.bar = self._apply_move_simulation(self.board, self.bar, move)
        if end == 'off':
            self.off[self.turn] += 1
            
        # 3. Check Win
        winner, pts = self.check_win()
        if winner != -1:
            self.score[winner] += pts
            self.phase = GamePhase.GAME_OVER
            self.winner = winner
            return pts, winner, True
            
        # 4. Check Turn End
        if not self.dice:
            # Turn Over
            self.turn = 1 - self.turn
            self.phase = GamePhase.DECIDE_CUBE_OR_ROLL
            self.legal_moves = []
        else:
            self.legal_moves = self._get_legal_moves_for_dice(tuple(self.dice))
            if not self.legal_moves:
                # No moves left for remaining dice -> Turn Over
                self.turn = 1 - self.turn
                self.phase = GamePhase.DECIDE_CUBE_OR_ROLL
                self.dice = []
                self.legal_moves = []
                
        return 0, -1, False

    def _roll_and_start_turn(self):
        self.current_roll = self.roll_dice()
        
        # Expand doubles
        d = list(self.current_roll)
        if d[0] == d[1]:
            d = [d[0]] * 4
        self.dice = d
        
        # For CPU (Full Turn Logic)
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
        dice = tuple(int(die) for die in roll)
        if not dice or any(die < 1 or die > 6 for die in dice):
            raise ValueError("roll must contain dice between 1 and 6")
        if len(dice) == 2 and dice[0] == dice[1]:
            dice = (dice[0],) * 4
        elif len(dice) > 4:
            raise ValueError("a turn cannot contain more than four dice")
        return self._get_legal_moves_for_dice(dice)

    def _get_legal_moves_for_dice(self, dice):
        labeled_paths = self._get_legal_labeled_paths(
            self.board, self.bar, tuple(dice)
        )
        sequences = {
            tuple(move for move, _die in path)
            for path in labeled_paths
            if path
        }
        return [
            list(sequence)
            for sequence in sorted(
                sequences,
                key=lambda sequence: tuple(
                    self._move_sort_key(move) for move in sequence
                ),
            )
        ]

    def _get_legal_labeled_paths(self, board, bar, dice):
        """Return maximally legal paths as ``((move, die), ...)`` tuples."""
        if not dice:
            return ()

        permutations = sorted(set(itertools.permutations(dice)))
        paths = []
        for ordered_dice in permutations:
            paths.extend(
                self._find_labeled_moves_recursive(
                    board.copy(), bar.copy(), ordered_dice
                )
            )

        max_moves = max((len(path) for path in paths), default=0)
        if max_moves == 0:
            return ()
        paths = [path for path in paths if len(path) == max_moves]

        if len(dice) == 2 and dice[0] != dice[1] and max_moves == 1:
            larger = max(dice)
            if any(path[0][1] == larger for path in paths):
                paths = [path for path in paths if path[0][1] == larger]

        return tuple(sorted(set(paths), key=lambda path: tuple(
            (*self._move_sort_key(move), die) for move, die in path
        )))

    def _find_labeled_moves_recursive(self, board, bar, dice):
        if not dice:
            return [()]

        die = dice[0]
        remaining_dice = dice[1:]
        moves = self._generate_single_moves(board, bar, die)
        if not moves:
            return self._find_labeled_moves_recursive(
                board, bar, remaining_dice
            )

        paths = []
        for move in moves:
            new_board, new_bar = self._apply_move_simulation(board, bar, move)
            for continuation in self._find_labeled_moves_recursive(
                new_board, new_bar, remaining_dice
            ):
                paths.append(((move, die),) + continuation)
        return paths

    def _get_move_distance(self, move):
        start, end = move
        player = self.turn
        
        if start == 'bar':
             # From Bar to End
             if player == 0:
                 # 24 -> end (e.g. 23). Dist = 1.
                 return 24 - end
             else:
                 # -1 -> end (e.g. 0). Dist = 1.
                 return end - (-1)
                 
        if end == 'off':
            # Distance from Start to "Virtual 0/-1"
            if player == 0:
                # Start index (e.g. 0 is 1 pip, 5 is 6 pips)
                return start + 1 
            else:
                # Start index (e.g. 23 is 1 pip, 18 is 6 pips)
                return 24 - start
                
        # Normal Move
        return abs(start - end)

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

    def get_afterstate(self, move_seq):
        """
        Simulates the entire move sequence and returns the resulting Game logic components.
        Returns: (board, bar, off, turn_switched_flag)
        
        NOTE: This does NOT switch the turn. 
        Evaluating 'Afterstate' usually means evaluating the board from the opponent's perspective (if turn switches)
        OR evaluating it from current perspective?
        Standard TD-Gammon: Output = P(Current Player Wins).
        If I move -> Board S'.
        Next is Opponent's turn.
        V(S') should be high if S' is good for ME.
        So we must check V(S') from MY perspective (Feature extraction must be aware of who is 'Me').
        """
        b = self.board.copy()
        ba = self.bar.copy()
        o = self.off.copy()
        player = self.turn
        
        for move in move_seq:
            if move[1] == 'off':
                o[player] += 1
            b, ba = self._apply_move_simulation(b, ba, move)
            
        return b, ba, o

    def check_win(self):
        """Checks if current player has borne off all checkers."""
        if self.off[0] >= 15:
            return 0, self._calculate_score(0)
        if self.off[1] >= 15:
            return 1, self._calculate_score(1)
        return -1, 0

    def get_win_type(self, winner: int) -> int:
        """
        Returns raw win type: 1 (Single), 2 (Gammon), 3 (Backgammon).
        Does NOT include cube multiplier.
        """
        loser = 1 - winner
        multiplier = 1
        
        if self.off[loser] == 0:
            multiplier = 2 # Gammon
            # Check Backgammon: Loser has checker on winner's home board or bar
            checkers_on_bar = self.bar[loser] > 0
            checkers_in_winner_home = False
            
            if winner == 0:
                # Winner is White (0-23). Home is 0-5.
                if any(x < 0 for x in self.board[0:6]): 
                    checkers_in_winner_home = True
            else:
                # Winner is Black. Home is 18-23.
                if any(x > 0 for x in self.board[18:24]):
                    checkers_in_winner_home = True
                    
            if checkers_on_bar or checkers_in_winner_home:
                multiplier = 3 # Backgammon
                
        return multiplier

    def _calculate_score(self, winner: int):
        """
        Calculates score multiplier (1, 2 for gammon, 3 for backgammon) * cube.
        """
        multiplier = self.get_win_type(winner)
        return self.cube_value * multiplier

    def get_pip_counts(self) -> Tuple[int, int]:
        """
        Returns (P0_Pips, P1_Pips).
        P0 moves 23->0 (Indices 0-5 are home). Corrected Logic check.
        Wait, earlier I analyzed P0 bears off from 0-5.
        So P0 moves HIGH to LOW. (23 -> 0).
        Actually...
        Original layout:
        P0 starts 24 (idx 0?). No.
        Let's look at Check Win again.
        "Winner is White (0-23). Home is 0-5. Loser is Black... pieces in 0-5"
        Logic verified: P0 moves Positive.
        If P0 moves Positive (0->23), then P0 Home is 18-23.
        Line 661: `if any(x < 0 for x in self.board[0:6]): checkers_in_winner_home` (Winner=0).
        If Winner=0, and their home is 18-23. Why check 0-6?
        The comment said "Winner is White... Home is 0-5".
        This implies P0 moves Negative (23->0).
        BUT _apply_move says: `if player == 0: b[start] -= 1`.
        Subtracting implies going to LOWER index? NO.
        `board` stores counts. `b[start] -= 1` means REMOVE piece.
        `b[end] += 1` means ADD piece.
        Move is `start -> end`.
        If Move is 24 -> 23.
        If logic is "Move by 1".
        Code: `dest = i + (die * direction)`. (Line 505).
        Direction?
        I need to check `_generate_single_moves` direction.
        Line 69: `self.direction = [1, -1]?` No.
        Let's find `direction`.
        If `player == 0`: direction = -1?
        I need to solve this Coordinate System ambiguity once and for all.
        """
        # Coordinate System Reverse Engineering:
        # P0 bears off at 0-5?
        # Check `_can_bear_off`. "Player 0: 0-5". 
        # If P0 home is 0-5, P0 moves towards 0.
        # So P0 moves [23 -> 0].
        # Distance calculation:
        # P0 Pip = Sum(count * (i + 1)).  (Index 0 needs 1 pip).
        
        # P1 Home is 18-23.
        # Check `_can_bear_off`. "Player 1: 18-23".
        # So P1 moves towards 24.
        # So P1 moves [0 -> 23].
        # Distance calculation:
        # P1 Pip = Sum(abs(count) * (24 - i)). (Index 23 needs 1 pip).
        
        p0_pip = 0
        p1_pip = 0
        
        for i in range(24):
            cnt = self.board[i]
            if cnt > 0: # P0
                # P0 Home 0-5. (move -1).
                # Pip = i + 1. (Index 0 is 1 pip away from off).
                p0_pip += cnt * (i + 1)
            elif cnt < 0: # P1
                # P1 Home 18-23. (move +1).
                # Pip = 24 - i. (Index 23 is 1 pip away from off).
                p1_pip += abs(cnt) * (24 - i)
                
        # Bar
        p0_pip += self.bar[0] * 25
        p1_pip += self.bar[1] * 25
        
        return p0_pip, p1_pip

    def reset_special_endgame(self):
        """
        Resets the game to a procedural "Late Game / Race" state.
        Now includes explicit "Bear Off" training (all checkers in home board).
        """
        self.board = [0] * 24
        self.bar = [0, 0]
        self.off = [0, 0]
        self.score = [0, 0] # Start fresh match
        self.turn = random.randint(0, 1)
        self.cube_value = 1
        self.cube_owner = -1
        self.crawford_active = False
        self.crawford_played = False
        
        # Scenario Selector:
        # 0: Pure Race (Contract 0-8)
        # 1: Bear Off (Contract 0-5)
        scenario = random.randint(0, 1)
        
        range_max = 8 if scenario == 0 else 5
        range_min_opp = 16 if scenario == 0 else 18
        
        # Distribute 15 checkers for P0 (Positive)
        p0_checkers = 15
        # Optional: In Bear Off, maybe some are already off?
        if scenario == 1 and random.random() < 0.3:
            already_off = random.randint(1, 5)
            self.off[0] += already_off
            p0_checkers -= already_off

        while p0_checkers > 0:
            idx = random.randint(0, range_max)
            self.board[idx] += 1
            p0_checkers -= 1
            
        # Distribute 15 checkers for P1 (Negative)
        p1_checkers = 15
        if scenario == 1 and random.random() < 0.3:
            already_off = random.randint(1, 5)
            self.off[1] += already_off
            p1_checkers -= already_off
            
        while p1_checkers > 0:
            # P1 Home is 18-23 (indices)
            # If race, 16-23. If bear off, 18-23.
            idx = random.randint(range_min_opp, 23)
            self.board[idx] -= 1
            p1_checkers -= 1
            
        self.phase = GamePhase.DECIDE_CUBE_OR_ROLL
        self.dice = []
        self.legal_moves = []
        self.winner = -1 # Explicit winner tracking

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


# --- UTILS ---
def get_obs_from_state(board, bar, off, perspective_player, score, cube_val, turn):
    """
    Generate egocentric observation for `perspective_player` (0 or 1).
    Returns basic 198 features + Score/Cube awareness injected by other layers if needed.
    """
    # Egocentric observation for `perspective_player`
    if perspective_player == 0:
        my_board = np.maximum(board, 0)
        opp_board = np.abs(np.minimum(board, 0))
    else:
        # P1 (Black/Negative) -> Flip board (0->23 becomes 23->0)
        my_board = np.abs(np.minimum(board, 0))[::-1]
        opp_board = np.maximum(board, 0)[::-1]
        
    features = []
    def encode_point(count):
        f = [0]*4
        if count >= 1: f[0] = 1
        if count >= 2: f[1] = 1
        if count >= 3: f[2] = 1
        if count > 3: f[3] = (count - 3) / 2.0
        return f
        
    for c in my_board: features.extend(encode_point(c))
    for c in opp_board: features.extend(encode_point(c))
    
    # Bar/Off
    my_bar = bar[perspective_player]
    opp_bar = bar[1-perspective_player]
    features.extend([my_bar/2.0, opp_bar/2.0])
    
    my_off = off[perspective_player]
    opp_off = off[1-perspective_player]
    features.extend([my_off/15.0, opp_off/15.0])
    
    # Turn: Always 1.0 (It's my turn in this state)
    features.append(1.0) 
    features.append(cube_val / 64.0)
    
    return np.array(features, dtype=np.float32)


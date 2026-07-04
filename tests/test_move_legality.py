import copy
import unittest

import numpy as np

from src.game import BackgammonGame, GamePhase


class PartialMoveLegalityTest(unittest.TestCase):
    def assert_game_state_equal(self, game, snapshot):
        np.testing.assert_array_equal(game.board, snapshot.board)
        self.assertEqual(game.bar, snapshot.bar)
        self.assertEqual(game.off, snapshot.off)
        self.assertEqual(game.dice, snapshot.dice)
        self.assertEqual(game.turn, snapshot.turn)
        self.assertEqual(game.phase, snapshot.phase)

    def test_blocked_destination_is_rejected_without_mutation(self):
        game = BackgammonGame()
        game.phase = GamePhase.DECIDE_MOVE
        game.dice = [1, 2]
        game.legal_moves = game.get_legal_moves((1, 2))
        snapshot = copy.deepcopy(game)

        with self.assertRaises(ValueError):
            game.step_partial((12, 11))

        self.assert_game_state_equal(game, snapshot)

    def test_bar_entry_has_priority(self):
        game = BackgammonGame()
        game.bar[0] = 1
        game.phase = GamePhase.DECIDE_MOVE
        game.dice = [1, 2]
        snapshot = copy.deepcopy(game)

        self.assertTrue(game.get_legal_partial_moves())
        self.assertTrue(
            all(move[0] == "bar" for move in game.get_legal_partial_moves())
        )
        with self.assertRaises(ValueError):
            game.step_partial((12, 10))

        self.assert_game_state_equal(game, snapshot)

    def test_larger_die_is_forced_when_only_one_die_can_enter(self):
        game = BackgammonGame()
        game.board = np.zeros(24, dtype=int)
        game.board[23] = -2  # Block entry with the 1.
        game.board[18] = 0   # Entry with the 6 is open.
        game.bar = [1, 0]
        game.off = [14, 13]
        game.turn = 0
        game.phase = GamePhase.DECIDE_MOVE
        game.dice = [1, 6]

        self.assertEqual(game.get_legal_partial_moves(), [("bar", 18)])
        game.step_partial(("bar", 18))
        self.assertEqual(game.board[18], 1)
        self.assertEqual(game.bar[0], 0)

    def test_smaller_die_is_allowed_when_larger_die_is_blocked(self):
        game = BackgammonGame()
        game.board = np.zeros(24, dtype=int)
        game.board[18] = -2
        game.bar = [1, 0]
        game.off = [14, 13]
        game.turn = 0
        game.phase = GamePhase.DECIDE_MOVE
        game.dice = [1, 6]

        self.assertEqual(game.get_legal_partial_moves(), [("bar", 23)])

    def test_legal_hit_updates_bar(self):
        game = BackgammonGame()
        game.board = np.zeros(24, dtype=int)
        game.board[5] = 1
        game.board[4] = -1
        game.off = [14, 14]
        game.turn = 0
        game.phase = GamePhase.DECIDE_MOVE
        game.dice = [1]

        game.step_partial((5, 4))

        self.assertEqual(game.board[4], 1)
        self.assertEqual(game.bar, [0, 1])
        self.assertEqual(game.turn, 1)

    def test_blocked_roll_returns_no_empty_sequence(self):
        game = BackgammonGame()
        game.board = np.zeros(24, dtype=int)
        game.board[23] = -2
        game.board[18] = -2
        game.bar = [1, 0]
        game.off = [14, 11]
        game.turn = 0

        self.assertEqual(game.get_legal_moves((1, 6)), [])


class FullMoveLegalityTest(unittest.TestCase):
    def test_out_of_range_action_is_rejected_without_mutation(self):
        game = BackgammonGame()
        game.phase = GamePhase.DECIDE_MOVE
        game.dice = [3, 1]
        game.legal_moves = game.get_legal_moves((3, 1))
        snapshot = copy.deepcopy(game)

        with self.assertRaises(ValueError):
            game.step(len(game.legal_moves))

        np.testing.assert_array_equal(game.board, snapshot.board)
        self.assertEqual(game.bar, snapshot.bar)
        self.assertEqual(game.off, snapshot.off)
        self.assertEqual(game.turn, snapshot.turn)

    def test_invalid_phase_actions_are_rejected(self):
        game = BackgammonGame()
        with self.assertRaises(ValueError):
            game.step(7)

        game.phase = GamePhase.RESPOND_TO_DOUBLE
        with self.assertRaises(ValueError):
            game.step(-1)


if __name__ == "__main__":
    unittest.main()

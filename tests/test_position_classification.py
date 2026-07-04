import unittest

import numpy as np

from src.game import BackgammonGame
from src.position_classification import PositionClass, PositionClassifier


class PositionClassifierTest(unittest.TestCase):
    def setUp(self):
        self.classifier = PositionClassifier()

    def classify(self, board, bar=(0, 0), off=(0, 0)):
        return self.classifier.classify(
            np.asarray(board, dtype=int), list(bar), list(off)
        )

    def empty_board(self):
        return np.zeros(24, dtype=int)

    def test_initial_position_is_contact(self):
        game = BackgammonGame()
        result = self.classifier.classify(game.board, game.bar, game.off)
        self.assertEqual(result.position_class, PositionClass.CONTACT)
        self.assertTrue(result.has_contact)

    def test_outer_race(self):
        board = self.empty_board()
        board[10] = 1
        board[15] = -1
        result = self.classify(board, off=(14, 14))
        self.assertEqual(result.position_class, PositionClass.RACE)
        self.assertFalse(result.has_contact)

    def test_one_sided_bearoff_coverage(self):
        board = self.empty_board()
        board[0] = 15
        board[23] = -15
        result = self.classify(board)
        self.assertEqual(
            result.position_class, PositionClass.ONE_SIDED_BEAROFF
        )
        self.assertEqual(result.player_home, (True, True))

    def test_two_sided_exact_coverage_boundary(self):
        board = self.empty_board()
        board[5] = 6
        board[18] = -6
        result = self.classify(board, off=(9, 9))
        self.assertEqual(
            result.position_class, PositionClass.TWO_SIDED_BEAROFF
        )

    def test_late_contact_is_crashed(self):
        board = self.empty_board()
        board[20] = 5
        board[3] = -5
        result = self.classify(board, off=(10, 10))
        self.assertEqual(result.position_class, PositionClass.CRASHED)
        self.assertTrue(result.has_contact)

    def test_buried_checkers_trigger_crashed_class(self):
        board = self.empty_board()
        board[0] = 11
        board[20] = 4
        board[3] = -15
        result = self.classify(board)
        self.assertEqual(result.position_class, PositionClass.CRASHED)
        self.assertEqual(result.buried_checkers[0], 9)

    def test_bar_means_contact(self):
        game = BackgammonGame()
        game.board[23] -= 1
        game.bar[0] = 1
        result = self.classifier.classify(game.board, game.bar, game.off)
        self.assertEqual(result.position_class, PositionClass.CONTACT)

    def test_game_over(self):
        board = self.empty_board()
        board[23] = -15
        result = self.classify(board, off=(15, 0))
        self.assertEqual(result.position_class, PositionClass.GAME_OVER)

    def test_player_swap_symmetry(self):
        game = BackgammonGame()
        original = self.classifier.classify(game.board, game.bar, game.off)
        mirrored_board = -game.board[::-1]
        mirrored = self.classifier.classify(
            mirrored_board,
            game.bar[::-1],
            game.off[::-1],
        )
        self.assertEqual(mirrored.position_class, original.position_class)
        self.assertEqual(
            mirrored.remaining_checkers,
            original.remaining_checkers[::-1],
        )
        self.assertEqual(
            mirrored.buried_checkers,
            original.buried_checkers[::-1],
        )

    def test_invalid_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            self.classify(np.zeros(23, dtype=int))


if __name__ == "__main__":
    unittest.main()

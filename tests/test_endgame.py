import unittest

import numpy as np

from src.endgame import BearoffEvaluator
from src.game import BackgammonGame


class EndgameEvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = BearoffEvaluator()

    def game_with_home_checkers(self, p0, p1, off0, off1, turn=0):
        game = BackgammonGame()
        game.board = np.zeros(24, dtype=int)
        for index, count in enumerate(p0):
            game.board[index] = count
        for index, count in enumerate(p1):
            game.board[23 - index] = -count
        game.bar = [0, 0]
        game.off = [off0, off1]
        game.turn = turn
        return game

    def test_home_state_orientation(self):
        game = self.game_with_home_checkers(
            (1, 2, 3, 0, 0, 0),
            (4, 1, 0, 0, 0, 0),
            9,
            10,
        )
        self.assertEqual(self.evaluator.home_state(game.board, 0), (1, 2, 3, 0, 0, 0))
        self.assertEqual(self.evaluator.home_state(game.board, 1), (4, 1, 0, 0, 0, 0))

    def test_exact_position_on_roll(self):
        game = self.game_with_home_checkers(
            (1, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0),
            14,
            14,
        )
        result = self.evaluator.evaluate_on_roll(game)
        self.assertIsNotNone(result)
        self.assertEqual(result["source"], "exact-two-sided-bearoff")
        self.assertEqual(result["win_prob"], 1.0)

    def test_rank_move_finds_immediate_bearoff(self):
        game = self.game_with_home_checkers(
            (1, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0),
            14,
            14,
        )
        game.legal_moves = game.get_legal_moves((1, 2))
        ranked = self.evaluator.rank_moves(game)
        self.assertTrue(ranked)
        self.assertEqual(ranked[0]["win_prob"], 1.0)
        self.assertEqual(ranked[0]["source"], "terminal-bearoff")


if __name__ == "__main__":
    unittest.main()

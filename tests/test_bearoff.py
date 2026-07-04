import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.bearoff import (
    OneSidedBearoffDatabase,
    generate_one_sided_database,
    legal_turn_afterstates,
    state_count,
)
from src.bearoff_two_sided import (
    TwoSidedBearoffDatabase,
    generate_two_sided_database,
)
from src.search import get_obs_gen5


class BearoffRulesTest(unittest.TestCase):
    def test_state_count(self):
        self.assertEqual(state_count(6, 15), 54_264)

    def test_gen5_score_features_match_training_contract(self):
        board = np.zeros(24, dtype=int)
        observation = get_obs_gen5(board, [0, 0], [0, 0], 0, [4, 2], 1, 0)
        np.testing.assert_allclose(observation[-2:], [0.8, 0.4])

    def test_terminal_state_is_stable(self):
        self.assertEqual(legal_turn_afterstates((0, 0), 3, 5), ((0, 0),))

    def test_oversized_die_only_bears_off_farthest_checker(self):
        afterstates = legal_turn_afterstates((1, 1, 0), 3, 3)
        self.assertEqual(afterstates, ((0, 0, 0),))

    def test_both_dice_are_used_when_possible(self):
        afterstates = legal_turn_afterstates((0, 0, 0, 0, 0, 1), 5, 6)
        self.assertEqual(afterstates, ((0, 0, 0, 0, 0, 0),))


class BearoffGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = generate_one_sided_database(points=3, checkers=4)

    def test_every_distribution_is_normalized(self):
        totals = self.result.probabilities.sum(axis=1)
        np.testing.assert_allclose(totals, 1.0, atol=1e-6)
        first_off_totals = self.result.first_off_probabilities.sum(axis=1)
        np.testing.assert_allclose(first_off_totals, 1.0, atol=1e-6)

    def test_terminal_and_single_checker_expectations(self):
        states = {
            tuple(map(int, state)): index
            for index, state in enumerate(self.result.states)
        }
        self.assertEqual(float(self.result.expected_rolls[states[(0, 0, 0)]]), 0.0)
        self.assertEqual(float(self.result.expected_rolls[states[(1, 0, 0)]]), 1.0)
        self.assertEqual(float(self.result.expected_rolls[states[(0, 0, 1)]]), 1.0)

    def test_round_trip_database(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test-bearoff.npz"
            self.result.save(path)
            database = OneSidedBearoffDatabase(path)

            self.assertEqual(database.metadata["policy"], self.result.metadata["policy"])
            self.assertAlmostEqual(database.expectation((1, 0, 0)), 1.0)
            self.assertEqual(database.best_afterstate((1, 0, 0), 1, 2), (0, 0, 0))

    def test_race_probability_respects_player_on_roll(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test-bearoff.npz"
            self.result.save(path)
            database = OneSidedBearoffDatabase(path)
            self.assertEqual(database.race_win_probability((1, 0, 0), (1, 0, 0)), 1.0)
            outcomes = database.race_outcomes((1, 0, 0), (1, 0, 0))
            self.assertEqual(outcomes["win"], 1.0)
            self.assertEqual(outcomes["win_gammon"], 0.0)


class TwoSidedBearoffTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = generate_two_sided_database(points=3, checkers=3)

    def test_terminal_values(self):
        self.assertTrue(np.all(self.result.win_probability[0, :] == 1.0))
        self.assertTrue(np.all(self.result.win_probability[1:, 0] == 0.0))

    def test_single_checker_race_favors_player_on_roll(self):
        states = {
            tuple(map(int, state)): index
            for index, state in enumerate(self.result.states)
        }
        one_checker = states[(1, 0, 0)]
        self.assertEqual(float(self.result.win_probability[one_checker, one_checker]), 1.0)

    def test_round_trip_and_best_move(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two-sided.npz"
            self.result.save(path)
            database = TwoSidedBearoffDatabase(path)
            self.assertEqual(database.equity((1, 0, 0), (1, 0, 0)), 1.0)
            self.assertEqual(
                database.best_afterstate((1, 0, 0), (1, 0, 0), 1, 2),
                (0, 0, 0),
            )


if __name__ == "__main__":
    unittest.main()

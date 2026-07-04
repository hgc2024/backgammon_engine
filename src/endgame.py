"""Phase-aware evaluation backed by the independent bear-off databases."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.bearoff import OneSidedBearoffDatabase
from src.bearoff_two_sided import TwoSidedBearoffDatabase
from src.position_classification import (
    DEFAULT_POSITION_CLASSIFIER,
    PositionClass,
)


class BearoffEvaluator:
    """Route covered home-board positions to deterministic tablebase logic."""

    def __init__(
        self,
        one_sided_path: str | Path | None = None,
        two_sided_path: str | Path | None = None,
    ):
        data_directory = Path(__file__).resolve().parents[1] / "data"
        one_path = Path(one_sided_path) if one_sided_path else (
            data_directory / "bearoff-one-sided-6x15-v2.npz"
        )
        two_path = Path(two_sided_path) if two_sided_path else (
            data_directory / "bearoff-two-sided-6x6-v1.npz"
        )

        self.one_sided = (
            OneSidedBearoffDatabase(one_path) if one_path.exists() else None
        )
        self.two_sided = (
            TwoSidedBearoffDatabase(two_path) if two_path.exists() else None
        )
        self.classifier = DEFAULT_POSITION_CLASSIFIER

    @staticmethod
    def is_home_board_bearoff(board: np.ndarray, bar: Sequence[int]) -> bool:
        homes = DEFAULT_POSITION_CLASSIFIER.player_home_status(board, bar)
        return homes[0] and homes[1]

    @staticmethod
    def home_state(board: np.ndarray, player: int) -> tuple[int, ...]:
        if player == 0:
            return tuple(int(max(board[index], 0)) for index in range(6))
        return tuple(int(max(-board[23 - index], 0)) for index in range(6))

    def _outcome(
        self,
        my_state: tuple[int, ...],
        opponent_state: tuple[int, ...],
        off: Sequence[int],
        player: int,
    ) -> dict[str, float | str] | None:
        if not any(my_state):
            return {
                "equity": 1.0,
                "win_prob": 1.0,
                "source": "terminal-bearoff",
            }

        no_gammon_possible = off[0] > 0 and off[1] > 0
        if (
            self.two_sided is not None
            and sum(my_state) <= 6
            and sum(opponent_state) <= 6
            and no_gammon_possible
        ):
            opponent_win = self.two_sided.equity(opponent_state, my_state)
            win = 1.0 - opponent_win
            return {
                "equity": 2.0 * win - 1.0,
                "win_prob": win,
                "source": "exact-two-sided-bearoff",
            }

        if self.one_sided is None:
            return None

        # The move has ended, so the opponent is now on roll. Invert their
        # outcome probabilities to recover the mover's result.
        opponent_outcomes = self.one_sided.race_outcomes(
            opponent_state, my_state
        )
        win = 1.0 - opponent_outcomes["win"]
        win_gammon = opponent_outcomes["lose_gammon"]
        lose_gammon = opponent_outcomes["win_gammon"]
        equity = 2.0 * win - 1.0 + win_gammon - lose_gammon
        return {
            "equity": equity,
            "win_prob": win,
            "win_gammon": win_gammon,
            "lose_gammon": lose_gammon,
            "source": "one-sided-race-distributions",
        }

    def rank_moves(self, game, moves=None) -> list[dict[str, object]]:
        """Rank legal bear-off moves, or return an empty list if uncovered."""
        if moves is None:
            moves = game.legal_moves
        classification = self.classifier.classify(
            game.board, game.bar, game.off
        )
        if not moves or classification.position_class not in (
            PositionClass.TWO_SIDED_BEAROFF,
            PositionClass.ONE_SIDED_BEAROFF,
        ):
            return []

        player = game.turn
        opponent = 1 - player
        results: list[dict[str, object]] = []

        for index, move in enumerate(moves):
            board, bar, off = game.get_afterstate(move)
            if bar[0] or bar[1]:
                return []
            my_state = self.home_state(board, player)
            opponent_state = self.home_state(board, opponent)
            outcome = self._outcome(my_state, opponent_state, off, player)
            if outcome is None:
                return []
            results.append({"index": index, "move": move, **outcome})

        results.sort(
            key=lambda item: (
                float(item["equity"]),
                float(item["win_prob"]),
                -int(item["index"]),
            ),
            reverse=True,
        )
        return results

    def evaluate_on_roll(self, game) -> dict[str, float | str] | None:
        """Evaluate a covered position before dice are rolled."""
        classification = self.classifier.classify(
            game.board, game.bar, game.off
        )
        if classification.position_class not in (
            PositionClass.GAME_OVER,
            PositionClass.TWO_SIDED_BEAROFF,
            PositionClass.ONE_SIDED_BEAROFF,
        ):
            return None

        player = game.turn
        opponent = 1 - player
        my_state = self.home_state(game.board, player)
        opponent_state = self.home_state(game.board, opponent)

        if not any(my_state):
            return {"equity": 1.0, "win_prob": 1.0, "source": "terminal-bearoff"}
        if not any(opponent_state):
            return {"equity": -1.0, "win_prob": 0.0, "source": "terminal-bearoff"}

        no_gammon_possible = game.off[0] > 0 and game.off[1] > 0
        if (
            self.two_sided is not None
            and sum(my_state) <= 6
            and sum(opponent_state) <= 6
            and no_gammon_possible
        ):
            win = self.two_sided.equity(my_state, opponent_state)
            return {
                "equity": 2.0 * win - 1.0,
                "win_prob": win,
                "source": "exact-two-sided-bearoff",
            }

        if self.one_sided is None:
            return None
        outcomes = self.one_sided.race_outcomes(my_state, opponent_state)
        equity = (
            2.0 * outcomes["win"]
            - 1.0
            + outcomes["win_gammon"]
            - outcomes["lose_gammon"]
        )
        return {
            "equity": equity,
            "win_prob": outcomes["win"],
            "source": "one-sided-race-distributions",
        }

"""Structural position classes for phase-aware backgammon evaluation.

The architecture follows the useful separation employed by mature backgammon
engines while using an independent classifier suited to this project's board
representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class PositionClass(str, Enum):
    GAME_OVER = "game_over"
    TWO_SIDED_BEAROFF = "two_sided_bearoff"
    ONE_SIDED_BEAROFF = "one_sided_bearoff"
    RACE = "race"
    CRASHED = "crashed"
    CONTACT = "contact"


@dataclass(frozen=True)
class PositionClassification:
    position_class: PositionClass
    has_contact: bool
    player_home: tuple[bool, bool]
    remaining_checkers: tuple[int, int]
    buried_checkers: tuple[int, int]


class PositionClassifier:
    """Classify standard backgammon positions without evaluating equity."""

    exact_bearoff_checkers = 6
    crashed_mobile_threshold = 6

    @staticmethod
    def _validate(board: np.ndarray, bar: Sequence[int], off: Sequence[int]) -> None:
        if np.asarray(board).shape != (24,):
            raise ValueError("board must contain exactly 24 points")
        if len(bar) != 2 or len(off) != 2:
            raise ValueError("bar and off must contain one count per player")
        if any(int(value) < 0 for value in bar) or any(
            int(value) < 0 for value in off
        ):
            raise ValueError("bar and off counts cannot be negative")

    @staticmethod
    def player_home_status(board: np.ndarray, bar: Sequence[int]) -> tuple[bool, bool]:
        """Return whether every remaining checker is in its home board."""
        p0_home = int(bar[0]) == 0 and not np.any(board[6:] > 0)
        p1_home = int(bar[1]) == 0 and not np.any(board[:18] < 0)
        return bool(p0_home), bool(p1_home)

    @staticmethod
    def remaining(board: np.ndarray, bar: Sequence[int]) -> tuple[int, int]:
        return (
            int(np.maximum(board, 0).sum()) + int(bar[0]),
            int(np.maximum(-board, 0).sum()) + int(bar[1]),
        )

    @staticmethod
    def contact_status(board: np.ndarray, bar: Sequence[int]) -> bool:
        """Return whether future blocking or hitting remains possible."""
        if int(bar[0]) or int(bar[1]):
            return True

        p0_points = np.flatnonzero(board > 0)
        p1_points = np.flatnonzero(board < 0)
        if not p0_points.size or not p1_points.size:
            return False

        # Player 0 travels high -> low; player 1 travels low -> high. Once
        # player 0's rearmost checker is below player 1's rearmost checker,
        # neither side can meet again.
        return bool(int(p0_points.max()) >= int(p1_points.min()))

    @staticmethod
    def buried(board: np.ndarray) -> tuple[int, int]:
        """Count excess checkers buried on the ace and deuce points.

        Two checkers retain a made point; additional checkers contribute
        little immediate mobility and are treated as structurally buried.
        """
        p0 = max(int(board[0]) - 2, 0) + max(int(board[1]) - 2, 0)
        p1 = max(int(-board[23]) - 2, 0) + max(int(-board[22]) - 2, 0)
        return p0, p1

    def classify(
        self, board: np.ndarray, bar: Sequence[int], off: Sequence[int]
    ) -> PositionClassification:
        board = np.asarray(board)
        self._validate(board, bar, off)

        remaining = self.remaining(board, bar)
        homes = self.player_home_status(board, bar)
        buried = self.buried(board)

        if int(off[0]) >= 15 or int(off[1]) >= 15 or 0 in remaining:
            return PositionClassification(
                PositionClass.GAME_OVER, False, homes, remaining, buried
            )

        if homes[0] and homes[1]:
            position_class = (
                PositionClass.TWO_SIDED_BEAROFF
                if max(remaining) <= self.exact_bearoff_checkers
                else PositionClass.ONE_SIDED_BEAROFF
            )
            return PositionClassification(
                position_class, False, homes, remaining, buried
            )

        contact = self.contact_status(board, bar)
        if not contact:
            return PositionClassification(
                PositionClass.RACE, False, homes, remaining, buried
            )

        mobile = (
            remaining[0] - buried[0],
            remaining[1] - buried[1],
        )
        crashed = any(
            count <= self.crashed_mobile_threshold for count in remaining
        ) or any(
            count <= self.crashed_mobile_threshold for count in mobile
        )
        return PositionClassification(
            PositionClass.CRASHED if crashed else PositionClass.CONTACT,
            True,
            homes,
            remaining,
            buried,
        )


DEFAULT_POSITION_CLASSIFIER = PositionClassifier()

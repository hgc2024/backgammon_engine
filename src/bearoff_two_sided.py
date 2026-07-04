"""Independent exact two-sided cubeless bear-off tablebases.

The recurrence is evaluated in increasing total pip count. For a position
``(on_roll, opponent)`` and a chosen legal afterstate ``next_state``:

* finishing immediately has value 1;
* otherwise the value is ``1 - V(opponent, next_state)``.

The player on roll chooses the afterstate with maximum value for each roll,
then the 21 dice classes are averaged using their 36-outcome weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.bearoff import ROLLS, ROLL_WEIGHTS, iter_states, legal_turn_afterstates


FORMAT_NAME = "backgammon-engine-two-sided-bearoff"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class TwoSidedGenerationResult:
    states: np.ndarray
    win_probability: np.ndarray
    metadata: dict[str, object]

    def save(self, destination: str | Path) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            states=self.states,
            win_probability=self.win_probability,
            rolls=np.asarray(ROLLS, dtype=np.uint8),
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )
        temporary.replace(path)
        return path


def generate_two_sided_database(
    points: int = 6,
    checkers: int = 6,
    progress: Callable[[int, int], None] | None = None,
) -> TwoSidedGenerationResult:
    """Generate exact cubeless equities for every covered state pair."""
    if points < 1 or points > 6:
        raise ValueError("this generator supports between 1 and 6 points")
    if checkers < 1 or checkers > 6:
        raise ValueError("two-sided generation supports up to 6 checkers")

    state_tuples = list(iter_states(points, checkers))
    state_index = {state: index for index, state in enumerate(state_tuples)}
    state_pips = np.asarray(
        [sum((point + 1) * count for point, count in enumerate(state)) for state in state_tuples],
        dtype=np.uint16,
    )
    max_pips = int(state_pips.max())
    states_by_pips: list[list[int]] = [[] for _ in range(max_pips + 1)]
    for index, pips in enumerate(state_pips):
        states_by_pips[int(pips)].append(index)

    successors: list[list[np.ndarray]] = []
    for state in state_tuples:
        state_successors: list[np.ndarray] = []
        for die_a, die_b in ROLLS:
            afterstates = legal_turn_afterstates(state, die_a, die_b)
            state_successors.append(
                np.asarray([state_index[afterstate] for afterstate in afterstates], dtype=np.uint16)
            )
        successors.append(state_successors)

    count = len(state_tuples)
    total_positions = count * count
    equity = np.full((count, count), np.nan, dtype=np.float64)

    # Terminal conventions. A normal lookup never asks for (0, 0), but defining
    # it makes the matrix complete and keeps validation straightforward.
    equity[0, :] = 1.0
    equity[1:, 0] = 0.0

    completed = 2 * count - 1
    for total_pips in range(1, 2 * max_pips + 1):
        minimum_us_pips = max(1, total_pips - max_pips)
        maximum_us_pips = min(max_pips, total_pips)

        for us_pips in range(minimum_us_pips, maximum_us_pips + 1):
            opponent_pips = total_pips - us_pips
            if opponent_pips < 1 or opponent_pips > max_pips:
                continue

            for us_index in states_by_pips[us_pips]:
                for opponent_index in states_by_pips[opponent_pips]:
                    expected_value = 0.0

                    for weight, candidate_indices in zip(
                        ROLL_WEIGHTS, successors[us_index]
                    ):
                        candidate_values = np.where(
                            candidate_indices == 0,
                            1.0,
                            1.0 - equity[opponent_index, candidate_indices],
                        )
                        if np.isnan(candidate_values).any():
                            raise RuntimeError(
                                "retrograde dependency was not generated before use"
                            )
                        expected_value += float(weight) * float(candidate_values.max())

                    equity[us_index, opponent_index] = expected_value / 36.0
                    completed += 1

        if progress:
            progress(completed, total_positions)

    if np.isnan(equity).any():
        raise RuntimeError("two-sided generation left uncovered positions")

    metadata: dict[str, object] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "points": points,
        "checkers_per_player": checkers,
        "one_sided_states": count,
        "position_pairs": total_positions,
        "equity": "cubeless probability that the player on roll wins",
        "state_order": "checker counts at distances 1..points",
        "license": "MIT",
        "provenance": "independently generated from backgammon rules",
    }
    return TwoSidedGenerationResult(
        states=np.asarray(state_tuples, dtype=np.uint8),
        win_probability=equity.astype(np.float32),
        metadata=metadata,
    )


class TwoSidedBearoffDatabase:
    """Read-only lookup interface for an exact two-sided tablebase."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as archive:
            self.states = archive["states"]
            self.win_probability = archive["win_probability"]
            self.rolls = archive["rolls"]
            self.metadata = json.loads(str(archive["metadata"].item()))

        if self.metadata.get("format") != FORMAT_NAME:
            raise ValueError(f"{self.path} is not a supported two-sided database")
        if self.metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported two-sided database format version")
        if tuple(map(tuple, self.rolls.tolist())) != ROLLS:
            raise ValueError("database dice ordering is incompatible")

        self._index = {
            tuple(int(value) for value in state): index
            for index, state in enumerate(self.states)
        }

    def index(self, state: Sequence[int]) -> int:
        normalized = tuple(int(value) for value in state)
        try:
            return self._index[normalized]
        except KeyError as exc:
            raise ValueError(f"state is outside this database: {normalized}") from exc

    def equity(self, on_roll: Sequence[int], opponent: Sequence[int]) -> float:
        """Return the exact cubeless win probability for the player on roll."""
        return float(
            self.win_probability[self.index(on_roll), self.index(opponent)]
        )

    def best_afterstate(
        self,
        on_roll: Sequence[int],
        opponent: Sequence[int],
        die_a: int,
        die_b: int,
    ) -> tuple[int, ...]:
        """Return the equity-optimal afterstate for a known dice roll."""
        state = tuple(int(value) for value in on_roll)
        opponent_index = self.index(opponent)
        candidates = legal_turn_afterstates(state, die_a, die_b)

        def value(candidate: tuple[int, ...]) -> tuple[float, tuple[int, ...]]:
            candidate_index = self.index(candidate)
            equity = (
                1.0
                if candidate_index == 0
                else 1.0 - float(self.win_probability[opponent_index, candidate_index])
            )
            return equity, tuple(-count for count in candidate)

        return max(candidates, key=value)

"""Independent one-sided backgammon bear-off tablebases.

The implementation in this module is derived from the rules of backgammon and
uses a project-specific file format.  It does not read or write GNU Backgammon
database files.

A state is a tuple of checker counts ordered by distance from bear-off:
``state[0]`` is the one-point, ``state[5]`` is the six-point.  Checkers not
present in the tuple have already been borne off.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np


FORMAT_NAME = "backgammon-engine-one-sided-bearoff"
FORMAT_VERSION = 2
POLICY_NAME = "minimum-expected-rolls-then-earliest-cdf"
ROLLS: tuple[tuple[int, int], ...] = tuple(
    (low, high) for low in range(1, 7) for high in range(low, 7)
)
ROLL_WEIGHTS = np.asarray(
    [1 if low == high else 2 for low, high in ROLLS], dtype=np.float64
)


def state_count(points: int, checkers: int) -> int:
    """Return the number of states with at most ``checkers`` on ``points``."""
    from math import comb

    if points < 1 or checkers < 0:
        raise ValueError("points must be positive and checkers non-negative")
    return comb(points + checkers, points)


def pip_count(state: Sequence[int]) -> int:
    """Return the number of pips remaining in a one-sided bear-off state."""
    return sum((index + 1) * count for index, count in enumerate(state))


def iter_states(points: int, checkers: int) -> Iterator[tuple[int, ...]]:
    """Yield every state, ordered so every legal successor appears first."""
    states: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...], remaining: int) -> None:
        if len(prefix) == points:
            states.append(prefix)
            return
        for count in range(remaining + 1):
            visit(prefix + (count,), remaining - count)

    visit((), checkers)
    states.sort(key=lambda state: (pip_count(state), sum(state), state))
    yield from states


def _single_die_successors(
    state: tuple[int, ...], die: int
) -> tuple[tuple[int, ...], ...]:
    """Return unique states reachable by legally playing one die."""
    successors: set[tuple[int, ...]] = set()

    for index, count in enumerate(state):
        if count == 0:
            continue

        distance = index + 1
        result = list(state)

        if die < distance:
            result[index] -= 1
            result[distance - die - 1] += 1
        elif die == distance:
            result[index] -= 1
        else:
            # An oversized die may bear off only a checker with none farther
            # from bear-off.
            if any(state[index + 1 :]):
                continue
            result[index] -= 1

        successors.add(tuple(result))

    return tuple(sorted(successors))


def _play_dice_in_order(
    state: tuple[int, ...], dice: tuple[int, ...]
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return ``(afterstate, used_dice)`` results for one fixed dice order."""
    frontier: set[tuple[tuple[int, ...], tuple[int, ...]]] = {(state, ())}

    for die in dice:
        next_frontier: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        for current, used in frontier:
            successors = _single_die_successors(current, die)
            if successors:
                next_frontier.update((successor, used + (die,)) for successor in successors)
            else:
                next_frontier.add((current, used))
        frontier = next_frontier

    return tuple(sorted(frontier))


@lru_cache(maxsize=262_144)
def legal_turn_afterstates(
    state: tuple[int, ...], die_a: int, die_b: int
) -> tuple[tuple[int, ...], ...]:
    """Return unique legal afterstates for a complete bear-off turn.

    The function enforces using the maximum possible number of dice and the
    larger-die rule when only one of two distinct dice can be played.
    """
    if not 1 <= die_a <= 6 or not 1 <= die_b <= 6:
        raise ValueError("dice must be between 1 and 6")
    if not any(state):
        return (state,)

    if die_a == die_b:
        results = _play_dice_in_order(state, (die_a,) * 4)
    else:
        results = (
            _play_dice_in_order(state, (die_a, die_b))
            + _play_dice_in_order(state, (die_b, die_a))
        )

    max_used = max(len(used) for _, used in results)
    results = tuple((after, used) for after, used in results if len(used) == max_used)

    if max_used == 1 and die_a != die_b:
        larger = max(die_a, die_b)
        if any(used == (larger,) for _, used in results):
            results = tuple((after, used) for after, used in results if used == (larger,))

    return tuple(sorted({after for after, _ in results}))


def _cdf_tiebreak(distribution: np.ndarray) -> tuple[float, ...]:
    """Create a key that prefers more probability mass on earlier finishes."""
    return tuple(-float(value) for value in np.cumsum(distribution))


@dataclass(frozen=True)
class GenerationResult:
    states: np.ndarray
    probabilities: np.ndarray
    first_off_probabilities: np.ndarray
    expected_rolls: np.ndarray
    best_next: np.ndarray
    metadata: dict[str, object]

    def save(self, destination: str | Path) -> Path:
        """Atomically write this database in the project-specific NPZ format."""
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            states=self.states,
            probabilities=self.probabilities,
            first_off_probabilities=self.first_off_probabilities,
            expected_rolls=self.expected_rolls,
            best_next=self.best_next,
            rolls=np.asarray(ROLLS, dtype=np.uint8),
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )
        temporary.replace(path)
        return path


def generate_one_sided_database(
    points: int = 6,
    checkers: int = 15,
    progress: Callable[[int, int], None] | None = None,
) -> GenerationResult:
    """Generate a one-sided completion-time tablebase by retrograde analysis.

    For each dice class, the policy minimizes expected rolls remaining.  Exact
    expectation ties prefer the distribution with the greatest cumulative
    chance of finishing earlier.
    """
    if points < 1 or points > 6:
        raise ValueError("this generator supports between 1 and 6 points")
    if checkers < 1 or checkers > 15:
        raise ValueError("this generator supports between 1 and 15 checkers")

    state_tuples = list(iter_states(points, checkers))
    total = len(state_tuples)
    expected_total = state_count(points, checkers)
    if total != expected_total:
        raise RuntimeError(f"state enumeration produced {total}, expected {expected_total}")

    index_by_state = {state: index for index, state in enumerate(state_tuples)}
    distributions: list[np.ndarray] = [np.asarray([1.0], dtype=np.float64)]
    first_off_distributions: list[np.ndarray] = [
        np.asarray([1.0], dtype=np.float64)
    ]
    expected = np.zeros(total, dtype=np.float64)
    best_next = np.zeros((total, len(ROLLS)), dtype=np.uint32)

    for state_index, state in enumerate(state_tuples[1:], start=1):
        roll_choices: list[int] = []
        roll_distributions: list[np.ndarray] = []

        for roll_index, (die_a, die_b) in enumerate(ROLLS):
            candidates = legal_turn_afterstates(state, die_a, die_b)
            candidate_indices = [index_by_state[candidate] for candidate in candidates]

            def candidate_key(candidate_index: int) -> tuple[object, ...]:
                return (
                    float(expected[candidate_index]),
                    _cdf_tiebreak(distributions[candidate_index]),
                    candidate_index,
                )

            chosen = min(candidate_indices, key=candidate_key)
            roll_choices.append(chosen)
            successor_distribution = distributions[chosen]
            shifted = np.zeros(successor_distribution.size + 1, dtype=np.float64)
            shifted[1:] = successor_distribution
            roll_distributions.append(shifted)
            best_next[state_index, roll_index] = chosen

        width = max(distribution.size for distribution in roll_distributions)
        distribution = np.zeros(width, dtype=np.float64)
        for weight, roll_distribution in zip(ROLL_WEIGHTS, roll_distributions):
            distribution[: roll_distribution.size] += weight * roll_distribution
        distribution /= 36.0

        # Remove only numerical zero tails; legitimate low-probability outcomes
        # remain represented.
        while distribution.size > 1 and distribution[-1] == 0.0:
            distribution = distribution[:-1]

        distributions.append(distribution)
        expected[state_index] = sum(
            turns * probability for turns, probability in enumerate(distribution)
        )

        if sum(state) < checkers:
            first_off_distribution = np.asarray([1.0], dtype=np.float64)
        else:
            roll_first_off: list[np.ndarray] = []
            for chosen in roll_choices:
                if sum(state_tuples[chosen]) < checkers:
                    shifted = np.zeros(2, dtype=np.float64)
                    shifted[1] = 1.0
                else:
                    successor_distribution = first_off_distributions[chosen]
                    shifted = np.zeros(successor_distribution.size + 1, dtype=np.float64)
                    shifted[1:] = successor_distribution
                roll_first_off.append(shifted)

            width = max(item.size for item in roll_first_off)
            first_off_distribution = np.zeros(width, dtype=np.float64)
            for weight, item in zip(ROLL_WEIGHTS, roll_first_off):
                first_off_distribution[: item.size] += weight * item
            first_off_distribution /= 36.0

        first_off_distributions.append(first_off_distribution)

        if progress and (state_index % 500 == 0 or state_index + 1 == total):
            progress(state_index + 1, total)

    max_turns = max(distribution.size for distribution in distributions) - 1
    probability_table = np.zeros((total, max_turns + 1), dtype=np.float32)
    for index, distribution in enumerate(distributions):
        probability_table[index, : distribution.size] = distribution

    max_first_off_turns = max(
        distribution.size for distribution in first_off_distributions
    ) - 1
    first_off_table = np.zeros(
        (total, max_first_off_turns + 1), dtype=np.float32
    )
    for index, distribution in enumerate(first_off_distributions):
        first_off_table[index, : distribution.size] = distribution

    states = np.asarray(state_tuples, dtype=np.uint8)
    metadata: dict[str, object] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "points": points,
        "checkers": checkers,
        "states": total,
        "max_turns": max_turns,
        "max_first_off_turns": max_first_off_turns,
        "policy": POLICY_NAME,
        "state_order": "checker counts at distances 1..points",
        "license": "MIT",
        "provenance": "independently generated from backgammon rules",
    }
    return GenerationResult(
        states=states,
        probabilities=probability_table,
        first_off_probabilities=first_off_table,
        expected_rolls=expected.astype(np.float32),
        best_next=best_next,
        metadata=metadata,
    )


class OneSidedBearoffDatabase:
    """Read-only interface to a generated one-sided bear-off database."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as archive:
            self.states = archive["states"]
            self.probabilities = archive["probabilities"]
            self.first_off_probabilities = archive["first_off_probabilities"]
            self.expected_rolls = archive["expected_rolls"]
            self.best_next = archive["best_next"]
            self.rolls = archive["rolls"]
            self.metadata = json.loads(str(archive["metadata"].item()))

        if self.metadata.get("format") != FORMAT_NAME:
            raise ValueError(f"{self.path} is not a supported bear-off database")
        if self.metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported bear-off database format version")
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

    def distribution(self, state: Sequence[int]) -> np.ndarray:
        """Return P(finish in exactly n rolls), indexed by n."""
        return self.probabilities[self.index(state)].copy()

    def expectation(self, state: Sequence[int]) -> float:
        return float(self.expected_rolls[self.index(state)])

    def race_win_probability(
        self, on_roll: Sequence[int], opponent: Sequence[int]
    ) -> float:
        """Return the cubeless race win chance under the one-sided policy.

        Both players use this database's minimum-expected-rolls policy. The
        player on roll wins ties because that player completes their bear-off
        before the opponent receives the corresponding turn.
        """
        on_roll_distribution = self.distribution(on_roll).astype(np.float64)
        opponent_distribution = self.distribution(opponent).astype(np.float64)
        opponent_survival = np.cumsum(opponent_distribution[::-1])[::-1]
        width = min(on_roll_distribution.size, opponent_survival.size)
        return float(
            np.dot(on_roll_distribution[:width], opponent_survival[:width])
        )

    def race_outcomes(
        self, on_roll: Sequence[int], opponent: Sequence[int]
    ) -> dict[str, float]:
        """Return cubeless win and gammon probabilities for a home-board race."""
        on_finish = self.distribution(on_roll).astype(np.float64)
        opponent_finish = self.distribution(opponent).astype(np.float64)
        on_first = self.first_off_probabilities[self.index(on_roll)].astype(np.float64)
        opponent_first = self.first_off_probabilities[self.index(opponent)].astype(
            np.float64
        )

        opponent_finish_survival = np.cumsum(opponent_finish[::-1])[::-1]
        width = min(on_finish.size, opponent_finish_survival.size)
        win = float(np.dot(on_finish[:width], opponent_finish_survival[:width]))

        opponent_first_survival = np.cumsum(opponent_first[::-1])[::-1]
        width = min(on_finish.size, opponent_first_survival.size)
        win_gammon = float(np.dot(on_finish[:width], opponent_first_survival[:width]))

        on_first_strict_survival = np.concatenate(
            (np.cumsum(on_first[::-1])[::-1][1:], np.asarray([0.0]))
        )
        width = min(opponent_finish.size, on_first_strict_survival.size)
        lose_gammon = float(
            np.dot(opponent_finish[:width], on_first_strict_survival[:width])
        )

        return {
            "win": win,
            "win_gammon": min(win, win_gammon),
            "lose_gammon": min(1.0 - win, lose_gammon),
        }

    def best_afterstate(
        self, state: Sequence[int], die_a: int, die_b: int
    ) -> tuple[int, ...]:
        roll = tuple(sorted((die_a, die_b)))
        try:
            roll_index = ROLLS.index(roll)
        except ValueError as exc:
            raise ValueError("dice must be between 1 and 6") from exc
        next_index = int(self.best_next[self.index(state), roll_index])
        return tuple(int(value) for value in self.states[next_index])

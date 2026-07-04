"""Generate the project's independent exact two-sided bear-off database."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from src.bearoff_two_sided import generate_two_sided_database


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an exact cubeless two-sided bear-off database."
    )
    parser.add_argument("--points", type=int, default=6)
    parser.add_argument("--checkers", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/bearoff-two-sided-6x6-v1.npz"),
    )
    args = parser.parse_args()

    started = time.perf_counter()
    last_report = 0.0

    def report(done: int, total: int) -> None:
        nonlocal last_report
        now = time.perf_counter()
        if now - last_report >= 1.0 or done == total:
            elapsed = now - started
            rate = done / elapsed if elapsed else 0.0
            print(
                f"\rGenerated {done:,}/{total:,} positions ({rate:,.0f}/s)",
                end="",
                flush=True,
            )
            last_report = now

    result = generate_two_sided_database(args.points, args.checkers, report)
    path = result.save(args.output)
    elapsed = time.perf_counter() - started
    print(
        f"\nSaved {result.metadata['position_pairs']:,} positions to {path} "
        f"in {elapsed:.1f}s ({path.stat().st_size / 1_048_576:.2f} MiB)."
    )


if __name__ == "__main__":
    main()

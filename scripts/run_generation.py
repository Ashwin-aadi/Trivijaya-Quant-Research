"""Generate the candidate corpus and record every attempt against the trial ledger.

Generation only. Auditing and backtesting are separate steps run afterwards, because generation
holds the GPU single-stream for hours and there is no reason to couple a long serial job to work
that can run on CPU later.

Every attempt increments the trial counter, successes and failures alike. The Deflated Sharpe
Ratio deflates by that count, so a ledger recording only the candidates that happened to compile
would understate the search and inflate every survivor.

Usage:
    python scripts/run_generation.py --n 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.audit.semantic import MODEL_TAG, is_available  # noqa: E402
from src.audit.stat import TrialCounter, default_counter_path  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.common.seeding import seed_everything  # noqa: E402
from src.generate.generator import generate_candidate  # noqa: E402
from src.generate.prompts import prompt_digest  # noqa: E402

_log = get_logger(__name__)

BASE_SEED = 42


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300, help="number of candidates to generate")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    # A smoke test still draws from the model, but it is not part of the corpus search and must not
    # land in the ledger the Deflated Sharpe denominator is read from.
    parser.add_argument("--ledger", type=Path, default=None,
                        help="trial ledger path; defaults to the project ledger")
    args = parser.parse_args()

    seed_everything(BASE_SEED)
    cfg = load_config()

    if not is_available():
        _log.error("Ollama is not reachable; generation cannot start")
        return 1

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or Path("runs") / stamp / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    counter = TrialCounter(args.ledger or default_counter_path(cfg))
    trials_before = counter.verify()
    _log.info("ledger verified at %d entries before this run", trials_before)
    _log.info("model %s, prompt digest %s", MODEL_TAG, prompt_digest()[:12])

    records: list[dict[str, object]] = []
    started = time.perf_counter()

    for index in range(args.n):
        candidate = generate_candidate(index, base_seed=BASE_SEED)
        # Recorded before anything else can go wrong with the file write: the trial happened
        # whether or not the artifact survives.
        counter.record(candidate.class_name, candidate.outcome)  # type: ignore[arg-type]

        path = out_dir / f"candidate_{index:03d}.py"
        if candidate.usable:
            path.write_text(candidate.source, encoding="utf-8")

        records.append({
            "index": index,
            "seed": candidate.seed,
            "theme": candidate.theme,
            "class_name": candidate.class_name,
            "attempts": candidate.attempts,
            "outcome": candidate.outcome,
            "path": str(path) if candidate.usable else None,
        })

        if (index + 1) % 10 == 0:
            rate = (time.perf_counter() - started) / (index + 1)
            remaining = rate * (args.n - index - 1) / 60
            usable = sum(1 for r in records if r["outcome"] == "evaluated")
            _log.info(
                "%d/%d generated, %d usable, %.1fs each, ~%.0f min remaining",
                index + 1, args.n, usable, rate, remaining,
            )

    elapsed = time.perf_counter() - started
    usable = sum(1 for r in records if r["outcome"] == "evaluated")
    summary = {
        "generated_at_utc": stamp,
        "model_tag": MODEL_TAG,
        "prompt_digest": prompt_digest(),
        "base_seed": BASE_SEED,
        "requested": args.n,
        "usable": usable,
        "trials_before": trials_before,
        "trials_after": counter.count(),
        "wall_clock_seconds": elapsed,
        "candidates": records,
    }
    (out_dir.parent / "generation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _log.info(
        "done: %d/%d usable in %.0f min; ledger %d -> %d",
        usable, args.n, elapsed / 60, trials_before, counter.count(),
    )
    print(f"corpus: {out_dir}")
    print(f"usable: {usable}/{args.n}   wall clock: {elapsed / 60:.0f} min")
    print(f"trial ledger: {trials_before} -> {counter.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

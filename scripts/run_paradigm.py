"""Generate one P4 arm's corpus, one arm per invocation, resumably.

P4 has six arms and roughly fifteen GPU hours of generation in it. Running them as one job would
mean a single interruption costs the whole batch, so this script does exactly one arm and can be
stopped and restarted without losing or repeating a draw.

**Resumption is by artefact, not by bookkeeping.** Draw ``i`` writes ``draw_0042.json`` and, when
usable, ``candidate_042.py``. A draw whose JSON already exists is skipped. Ctrl-C at any moment is
therefore safe: the worst case is that the draw in flight is lost and redrawn, and because seeding
is deterministic the redraw is the same draw.

**Every arm has its own trial ledger.** The PI ruled on 2026-08-04 that per-arm deflation is the
primary analysis and pooled deflation the sensitivity analysis; pooling is the concatenation of
these files and is computed at analysis time. **P1's project ledger is not touched** — it stands at
the count AlphaAudit's published results were deflated against, and appending P4's draws to it would
retrospectively change a released paper's N.

**The ledger increments by candidates evaluated, not by draws.** A tree search that completes
twelve designs to return one strategy has searched twelve times, and RQ4 is the question of what
that honest count does to it.

Usage (Windows):
    .venv\\Scripts\\python.exe scripts/run_paradigm.py --arm G1 --n 120
    .venv\\Scripts\\python.exe scripts/run_paradigm.py --arm G7 --n 120 --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.audit.semantic import MODEL_TAG, is_available  # noqa: E402
from src.audit.stat import TrialCounter  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.common.seeding import seed_everything  # noqa: E402
from src.generate.paradigms.registry import ARMS, build  # noqa: E402
from src.generate.prompts import prompt_digest  # noqa: E402
from src.generate.tokens import TokenAccount  # noqa: E402

_log = get_logger(__name__)

BASE_SEED = 42
CORPUS_ROOT = Path("benchmarks/generationbench/corpus")


def arm_dir(arm: str) -> Path:
    return CORPUS_ROOT / arm


def ledger_path(arm: str) -> Path:
    return CORPUS_ROOT / arm / "trial_ledger.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS),
                        help="which arm to generate")
    parser.add_argument("--n", type=int, default=120,
                        help="draws in the arm; the pre-registered figure is 120")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many new draws this sitting; resume by rerunning")
    args = parser.parse_args()

    seed_everything(BASE_SEED)
    cfg = load_config()
    if not is_available():
        _log.error("Ollama is not reachable; generation cannot start")
        return 1

    with RunManifest(cfg, f"run_paradigm.py --arm {args.arm}") as manifest:
        manifest.add_model(MODEL_TAG)
        summary = _generate(args)
        for key, value in summary.items():
            manifest.note(key, value)
    return 0


def _generate(args: argparse.Namespace) -> dict[str, Any]:
    """Draw the arm, resuming from whatever is already on disk. Returns the run summary."""
    out = arm_dir(args.arm)
    out.mkdir(parents=True, exist_ok=True)
    full_name = ARMS[args.arm]

    counter = TrialCounter(ledger_path(args.arm))
    # Verified, not counted. A ledger that has been edited must stop the run rather than quietly
    # supply a smaller N to the deflation later.
    trials_before = counter.verify()

    paradigm, fitness = build(args.arm)
    account = TokenAccount()
    _log.info("arm %s (%s), n=%d, model %s, prompt digest %s",
              args.arm, full_name, args.n, MODEL_TAG, prompt_digest()[:12])
    _log.info("ledger verified at %d entries", trials_before)

    started = time.perf_counter()
    done_this_sitting = 0
    try:
        for index in range(args.n):
            record_path = out / f"draw_{index:04d}.json"
            if record_path.exists():
                continue
            if args.limit is not None and done_this_sitting >= args.limit:
                _log.info("reached --limit %d; rerun the same command to continue", args.limit)
                break

            draw_started = time.perf_counter()
            draw = paradigm.draw(index, base_seed=BASE_SEED, account=account)
            elapsed = time.perf_counter() - draw_started

            # Recorded before the artefact is written: the trials happened whether or not anything
            # survives them. One entry per candidate the draw evaluated.
            outcomes = draw.attempt_outcomes or (draw.outcome,)
            for n, outcome in enumerate(outcomes, start=1):
                label = f"{full_name}#{index:04d}#{n:02d}"
                counter.record(label, outcome)  # type: ignore[arg-type]

            if draw.usable:
                (out / f"candidate_{index:03d}.py").write_text(draw.source, encoding="utf-8")

            payload: dict[str, Any] = asdict(draw)
            payload["seconds"] = elapsed
            payload["model_tag"] = MODEL_TAG
            payload["prompt_digest"] = prompt_digest()
            payload["base_seed"] = BASE_SEED
            payload["trials_after"] = counter.count()
            record_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            done_this_sitting += 1
            remaining = args.n - index - 1
            rate = (time.perf_counter() - started) / done_this_sitting
            _log.info(
                "%s draw %d/%d: %s, %d calls, %d output tokens, %.0fs "
                "(%.0f min left at this rate)",
                args.arm, index + 1, args.n, draw.outcome, draw.calls,
                draw.usage.output_tokens, elapsed, rate * remaining / 60,
            )
    except KeyboardInterrupt:
        _log.warning("interrupted; %d draws completed this sitting and are on disk",
                     done_this_sitting)
    finally:
        if fitness is not None and hasattr(fitness, "close"):
            fitness.close()

    total_elapsed = time.perf_counter() - started
    completed = sorted(out.glob("draw_*.json"))
    usable = sum(
        1 for p in completed
        if json.loads(p.read_text(encoding="utf-8"))["outcome"] == "evaluated"
    )

    summary = {
        "arm": args.arm,
        "paradigm": full_name,
        "updated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "requested": args.n,
        "completed": len(completed),
        "usable": usable,
        "trials_before_this_sitting": trials_before,
        "trials_after": counter.count(),
        "seconds_this_sitting": total_elapsed,
        "draws_this_sitting": done_this_sitting,
        "tokens": account.to_dict(),
        "fitness": fitness.stats() if fitness is not None and hasattr(fitness, "stats") else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"arm {args.arm}: {len(completed)}/{args.n} draws on disk, {usable} usable")
    print(f"this sitting: {done_this_sitting} draws in {total_elapsed / 60:.1f} min")
    print(f"trial ledger: {trials_before} -> {counter.count()}")
    if len(completed) < args.n:
        print(f"NOT FINISHED - rerun: .venv\\Scripts\\python.exe scripts/run_paradigm.py "
              f"--arm {args.arm} --n {args.n}")
    return summary


if __name__ == "__main__":
    raise SystemExit(main())

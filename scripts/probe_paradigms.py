"""Time a few draws from every arm and project the full run, before committing to it.

Required by CLAUDE.md before Phase 4.1 generation begins: *"time ten generations per paradigm,
project the full run, and report the estimate. If it exceeds a working week of unattended compute,
halt and propose a reduced sample size rather than starting a run that cannot finish."*

**The probe does not touch the corpus or the trial ledger.** It writes to a throwaway directory
under `runs/` with its own ledger, following the precedent set for P1's smoke test. Because seeding
is deterministic, the draws it makes are the same draws the real run will make at those indices —
the probe measures the real thing rather than a proxy, and costs the same compute twice on purpose
so the estimate is honest.

**G7 is the expensive one and the one worth probing carefully.** It completes a full strategy per
iteration and backtests each one, so its per-draw time includes CPU that the other arms do not
spend. Probing it with fewer draws than the others is a false economy.

Usage (Windows):
    .venv\\Scripts\\python.exe scripts/probe_paradigms.py --draws 2
    .venv\\Scripts\\python.exe scripts/probe_paradigms.py --draws 2 --arm G7
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
from src.common.log import get_logger  # noqa: E402
from src.common.seeding import seed_everything  # noqa: E402
from src.generate.paradigms.registry import ARMS, build  # noqa: E402
from src.generate.tokens import TokenAccount  # noqa: E402

_log = get_logger(__name__)

BASE_SEED = 42
TARGET_N = 120


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=2, help="probe draws per arm")
    parser.add_argument("--arm", choices=sorted(ARMS), default=None, help="probe one arm only")
    parser.add_argument("--n", type=int, default=TARGET_N, help="the run being projected")
    args = parser.parse_args()

    seed_everything(BASE_SEED)
    if not is_available():
        _log.error("Ollama is not reachable")
        return 1

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path("runs") / f"probe_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for short in [args.arm] if args.arm else list(ARMS):
        paradigm, fitness = build(short)
        account = TokenAccount()
        started = time.perf_counter()
        usable = 0
        try:
            for index in range(args.draws):
                draw = paradigm.draw(index, base_seed=BASE_SEED, account=account)
                usable += int(draw.usable)
                _log.info("%s draw %d: %s, %d calls, %d output tokens",
                          short, index, draw.outcome, draw.calls, draw.usage.output_tokens)
        finally:
            if fitness is not None and hasattr(fitness, "close"):
                fitness.close()
        elapsed = time.perf_counter() - started
        full = ARMS[short]
        rows.append({
            "arm": short,
            "paradigm": full,
            "draws": args.draws,
            "usable": usable,
            "seconds_per_draw": elapsed / max(args.draws, 1),
            "output_tokens_per_draw": account.usage[full].output_tokens / max(args.draws, 1),
            "calls_per_draw": account.calls[full] / max(args.draws, 1),
            "projected_hours": elapsed / max(args.draws, 1) * args.n / 3600,
        })

    print(f"\n{'arm':<5} {'s/draw':>9} {'tok/draw':>10} {'calls':>7} "
          f"{'hours at n=' + str(args.n):>16}")
    print("-" * 52)
    for row in rows:
        print(f"{row['arm']:<5} {row['seconds_per_draw']:>9.1f} "
              f"{row['output_tokens_per_draw']:>10.0f} {row['calls_per_draw']:>7.1f} "
              f"{row['projected_hours']:>16.1f}")
    total = sum(float(r["projected_hours"]) for r in rows)
    print("-" * 52)
    print(f"{'total':<5} {'':>9} {'':>10} {'':>7} {total:>16.1f}")

    (out / "probe.json").write_text(
        json.dumps({"model_tag": MODEL_TAG, "n_projected": args.n,
                    "total_projected_hours": total, "arms": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwritten to {out / 'probe.json'}")
    print("A total far above the Checkpoint 4.0 estimate is a reason to halt and re-plan n,")
    print("not a reason to start the run and hope.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

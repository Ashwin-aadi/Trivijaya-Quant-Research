"""Show how far P4's generation has got, across every arm. Read-only.

Safe to run at any time, including while an arm is generating: it opens nothing for writing and
holds no lock. Intended to be the command the PI runs in a second terminal.

Usage (Windows):
    .venv\\Scripts\\python.exe scripts/paradigm_status.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.generate.paradigms.registry import ARMS  # noqa: E402

CORPUS_ROOT = Path("benchmarks/generationbench/corpus")
TARGET = 120


def _read(path: Path) -> dict[str, object] | None:
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        # A draw file being written right now is not an error; it is a file that will exist in a
        # second. Reporting it as corrupt would send the PI looking for a problem that is not there.
        return None


def main() -> int:
    print(f"{'arm':<5} {'paradigm':<24} {'draws':>9} {'usable':>7} {'yield':>7} "
          f"{'out tok':>10} {'sec/draw':>9} {'est left':>9}")
    print("-" * 88)

    total_done = 0
    total_target = 0
    for short, full in ARMS.items():
        arm = CORPUS_ROOT / short
        records = [r for r in (_read(p) for p in sorted(arm.glob("draw_*.json"))) if r]
        done = len(records)
        total_done += done
        total_target += TARGET

        if not records:
            print(f"{short:<5} {full:<24} {'0/' + str(TARGET):>9} {'-':>7} {'-':>7} "
                  f"{'-':>10} {'-':>9} {'not started':>11}")
            continue

        usable = sum(1 for r in records if r.get("outcome") == "evaluated")
        tokens = sum(int(dict(r["usage"])["output_tokens"]) for r in records)  # type: ignore[arg-type]
        seconds = sum(float(r.get("seconds", 0.0)) for r in records)  # type: ignore[arg-type]
        per_draw = seconds / done
        left_min = per_draw * max(TARGET - done, 0) / 60
        print(f"{short:<5} {full:<24} {f'{done}/{TARGET}':>9} {usable:>7} "
              f"{usable / done:>6.1%} {tokens:>10,} {per_draw:>9.1f} {left_min:>7.0f}m")

    print("-" * 88)
    print(f"{'total':<5} {'':<24} {f'{total_done}/{total_target}':>9}")

    missing = [s for s in ARMS if len(list((CORPUS_ROOT / s).glob("draw_*.json"))) < TARGET]
    if missing:
        print("\nnot yet complete: " + ", ".join(missing))
        print("next: .venv\\Scripts\\python.exe scripts/run_paradigm.py --arm "
              f"{missing[0]} --n {TARGET}")
    else:
        print("\nall six arms complete. Next: verify_corpus.py, then halt at Checkpoint 4.1.")
    # Nothing here is a pass/fail judgement, so the exit code is always 0. `verify_corpus.py` is
    # the script that is allowed to fail.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

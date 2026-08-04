"""Check P4's generated corpus for the failures that would invalidate the experiment silently.

Every check here is one that fails *loudly* rather than producing a plausible-looking wrong number.
That is the point: a corpus with a broken token count still yields a comparison table, and the table
would be wrong in a way nobody could see. Run this after each arm finishes, and again before any
analysis.

**What it checks, and why each one matters if it fails:**

1. **Draw indices are complete and unique.** A missing index is a draw that silently left the yield
   denominator; a duplicate is one counted twice.
2. **Every draw's theme matches ``theme_for(index)``.** The arms are compared on identical tasks. A
   theme mismatch means one arm answered a different question and the comparison is void.
3. **The prompt digest matches the frozen P1 specification.** A changed prompt makes the arm
   incomparable to P1's corpus, which is the compute-matched control.
4. **The model tag is identical across every draw and every arm.** P4 holds the model fixed and
   varies only the methodology; two model tags in one corpus destroys that.
5. **Every draw generated a non-zero number of output tokens.** A zero means Ollama returned no
   counts, which means RULE 11 cannot be enforced for that draw and the compute matching is
   fictional.
6. **Usable draws have a file on disk whose text is exactly the recorded source**, and that file
   still passes P1's conformance rule.
7. **Unusable draws have no file**, so nothing unaudited can leak into a backtest.
8. **The trial ledger's hash chain verifies**, and its entry count equals the number of candidates
   the draws say they evaluated. A ledger short of that undercounts the search and every Deflated
   Sharpe Ratio computed from it is too generous.

Usage (Windows):
    .venv\\Scripts\\python.exe scripts/verify_corpus.py
    .venv\\Scripts\\python.exe scripts/verify_corpus.py --arm G7

Exits 0 when every check passes, 1 when any fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.audit.semantic import MODEL_TAG  # noqa: E402
from src.audit.stat import TrialCounter  # noqa: E402
from src.generate.generator import _conformance_failure  # noqa: E402
from src.generate.paradigms.registry import ARMS  # noqa: E402
from src.generate.prompts import prompt_digest, theme_for  # noqa: E402

CORPUS_ROOT = Path("benchmarks/generationbench/corpus")

#: The P1 task specification's digest, recorded in `benchmarks/generationbench/PREREGISTRATION.md`
#: §2 before any data existed. Hardcoded here so a change to `src/generate/prompts.py` fails this
#: check rather than being blessed by whatever the file happens to hash to today.
FROZEN_PROMPT_DIGEST = "f307433c7bda8595d52432b3bcb4f723663bfe706112a41f84e4beacfbde9934"


def _check_arm(short: str, expected_n: int, failures: list[str]) -> None:
    arm = CORPUS_ROOT / short
    full = ARMS[short]
    paths = sorted(arm.glob("draw_*.json"))
    if not paths:
        print(f"{short:<4} not started")
        return

    draws: dict[int, dict[str, Any]] = {}
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        index = int(record["index"])
        if index in draws:
            failures.append(f"{short}: draw index {index} appears twice")
        draws[index] = record

    missing = [i for i in range(expected_n) if i not in draws]
    if missing:
        failures.append(
            f"{short}: {len(missing)} of {expected_n} draws missing "
            f"(first: {missing[0]}) - incomplete, not invalid"
        )

    candidates_evaluated = 0
    for index, record in sorted(draws.items()):
        where = f"{short} draw {index}"
        if record["paradigm"] != full:
            failures.append(f"{where}: paradigm is {record['paradigm']!r}, expected {full!r}")
        if record["theme"] != theme_for(index):
            failures.append(f"{where}: theme does not match theme_for({index})")
        if record.get("prompt_digest") != FROZEN_PROMPT_DIGEST:
            failures.append(f"{where}: prompt digest is not the frozen P1 specification")
        if record.get("model_tag") != MODEL_TAG:
            failures.append(f"{where}: model tag is {record.get('model_tag')!r}")
        if int(record["usage"]["output_tokens"]) <= 0:
            failures.append(f"{where}: zero output tokens recorded - RULE 11 unenforceable")
        candidates_evaluated += int(record.get("candidates_evaluated", 1))

        py = arm / f"candidate_{index:03d}.py"
        if record["outcome"] == "evaluated":
            if not py.exists():
                failures.append(f"{where}: usable but no candidate file on disk")
            elif py.read_text(encoding="utf-8") != record["source"]:
                failures.append(f"{where}: candidate file differs from the recorded source")
            else:
                reason = _conformance_failure(record["source"])
                if reason is not None:
                    failures.append(f"{where}: recorded usable but fails conformance: {reason}")
        elif py.exists():
            failures.append(f"{where}: not usable but a candidate file exists")

    ledger = TrialCounter(arm / "trial_ledger.jsonl")
    try:
        verified = ledger.verify()
    except Exception as exc:  # noqa: BLE001 - a broken chain is the finding, not a crash
        failures.append(f"{short}: trial ledger does not verify: {exc}")
        verified = -1

    attempts = sum(len(r.get("attempt_outcomes") or (r["outcome"],)) for r in draws.values())
    if verified >= 0 and verified != attempts:
        failures.append(
            f"{short}: ledger has {verified} entries but the draws record {attempts} attempts"
        )

    usable = sum(1 for r in draws.values() if r["outcome"] == "evaluated")
    print(
        f"{short:<4} {len(draws):>4}/{expected_n} draws  {usable:>4} usable  "
        f"{candidates_evaluated:>5} candidates evaluated  {verified:>5} ledger entries"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), default=None, help="check one arm only")
    parser.add_argument("--n", type=int, default=120, help="pre-registered draws per arm")
    args = parser.parse_args()

    failures: list[str] = []
    if prompt_digest() != FROZEN_PROMPT_DIGEST:
        failures.append(
            "src/generate/prompts.py no longer hashes to the digest pre-registered in "
            "PREREGISTRATION.md §2. The task specification has changed and P4 is no longer "
            "comparable to P1."
        )

    for short in [args.arm] if args.arm else list(ARMS):
        _check_arm(short, args.n, failures)

    print()
    if failures:
        print(f"FAILED - {len(failures)} problem(s):")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

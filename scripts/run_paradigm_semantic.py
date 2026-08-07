"""Semantic audit over the P4 corpus, resumable at candidate granularity.

`run_corpus_audit.py` runs the same layer but writes only once, at the end: killing it three hours
in loses three hours. This corpus is 2,495 candidates against P1's 1,550 and the PI needs to be able
to stop the machine at any moment, so verdicts are appended to a JSONL sidecar as they are produced
and an interrupted run resumes from the last line written.

**The layer itself is untouched.** `classify` and `extract_rationale` are imported from the frozen
auditor and P1's driver respectively; nothing here decides anything a verdict depends on. Only the
write schedule differs, and a verdict is a function of one candidate alone, so the order candidates
are scored in cannot change any of them.

Usage:
    python scripts/run_paradigm_semantic.py              # every arm, resuming
    python scripts/run_paradigm_semantic.py --arm G1     # one arm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from scripts.run_corpus_audit import extract_rationale  # noqa: E402

from src.audit.semantic import classify, is_available  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.generate.paradigms.registry import ARMS  # noqa: E402

_log = get_logger(__name__)

CORPUS = Path("benchmarks/generationbench/corpus")

#: Measured over 300 candidates in `runs/audit_log_b1.txt`, stable at 4.3-4.4 s throughout.
SECONDS_PER_CANDIDATE = 4.35


def _done(sidecar: Path) -> dict[str, dict[str, Any]]:
    """Verdicts already on disk, keyed by candidate stem.

    A truncated final line is the normal result of killing the process mid-write, so it is dropped
    and that one candidate is rescored rather than treated as corruption of the whole file.
    """
    if not sidecar.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[record["name"]] = record
    return out


def _score(path: Path) -> dict[str, Any]:
    """One candidate's semantic verdict. A model failure is recorded as a datum, never raised."""
    source = path.read_text(encoding="utf-8")
    try:
        finding = classify(extract_rationale(source), source)
    except Exception as exc:  # noqa: BLE001 - matches P1's driver exactly
        return {"name": path.stem, "rejected": False, "label": "error",
                "confidence": 0.0, "error": f"{type(exc).__name__}: {exc}"[:200]}
    return {"name": path.stem, "rejected": bool(finding.is_defect), "label": finding.label,
            "confidence": float(finding.confidence), "error": None}


def run_arm(arm: str) -> int:
    """Score every unscored candidate in `arm`, appending each verdict before starting the next."""
    paths = sorted((CORPUS / arm).glob("candidate_*.py"))
    sidecar = CORPUS / arm / "semantic_verdicts.jsonl"
    done = _done(sidecar)
    pending = [p for p in paths if p.stem not in done]
    if not pending:
        _log.info("%s: all %d already scored", arm, len(paths))
        return 0

    _log.info("%s: %d candidates, %d already scored, %d pending, ~%.0f min",
              arm, len(paths), len(done), len(pending),
              len(pending) * SECONDS_PER_CANDIDATE / 60)

    started = time.perf_counter()
    with sidecar.open("a", encoding="utf-8") as handle:
        for index, path in enumerate(pending, start=1):
            handle.write(json.dumps(_score(path), sort_keys=True) + "\n")
            handle.flush()  # the point of the exercise: survive a kill between candidates
            if index % 25 == 0:
                rate = (time.perf_counter() - started) / index
                _log.info("%s semantic %d/%d, %.1fs each, ~%.0f min left", arm, index,
                          len(pending), rate, rate * (len(pending) - index) / 60)
    return 0


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), default=None,
                        help="one arm; default is every arm in registry order")
    args = parser.parse_args()

    if not is_available():
        _log.error("Ollama unreachable; semantic layer cannot run")
        return 1

    arms = [args.arm] if args.arm else list(ARMS)
    total = sum(len(sorted((CORPUS / a).glob("candidate_*.py"))) for a in arms)
    _log.info("%d candidates across %s; full-run estimate %.1f h",
              total, ",".join(arms), total * SECONDS_PER_CANDIDATE / 3600)

    for arm in arms:
        run_arm(arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())

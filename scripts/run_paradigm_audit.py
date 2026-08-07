"""Static and statistical audit of the P4 corpus, one arm at a time, per-arm and pooled.

P1's driver cannot be pointed at this corpus. It deflates against
``data/processed/trial_ledger.jsonl`` -- AlphaAudit's project ledger -- and adds a correction
constant specific to P1's first generation run. Amendment 1.2 rules that each P4 arm deflates
against its own hash-chained ledger at ``corpus/<arm>/trial_ledger.jsonl``, that per-arm is the
primary analysis and pooled the sensitivity, and that **P1's ledger is not touched**: appending P4's
draws would retrospectively change a released paper's N.

**Nothing about the layers themselves differs.** `static_layer` and `statistical_layer` are imported
from P1's driver, so an arm's verdicts are produced by the same code that produced AlphaAudit's. The
only new decision here is which N goes into the deflation, and that decision was made by the PI in
advance, in writing.

The semantic layer is not run here -- it holds the GPU for three hours and lives in
``run_paradigm_semantic.py``, which is resumable. Whatever verdicts that has written so far are
merged in, and the coverage is recorded so a partial run is never mistaken for a complete one.

Writes ``benchmarks/generationbench/corpus/<arm>/audit_results.json``.

Usage:
    python scripts/run_paradigm_audit.py
    python scripts/run_paradigm_audit.py --arm G1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from scripts.run_corpus_audit import static_layer, statistical_layer  # noqa: E402

from src.audit.stat import TrialCounter  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.generate.paradigms.registry import ARMS  # noqa: E402

_log = get_logger(__name__)

CORPUS = Path("benchmarks/generationbench/corpus")


def arm_trials(arm: str) -> int:
    """The arm's honest trial count, from its own hash-chained ledger.

    `verify()` raises if the chain does not reconstruct, which is the intended behaviour: a corpus
    whose trial count cannot be trusted must not produce a deflated Sharpe at all.
    """
    return TrialCounter(CORPUS / arm / "trial_ledger.jsonl").verify()


def semantic_so_far(arm: str) -> dict[str, dict[str, Any]]:
    """Semantic verdicts written by the resumable runner, however many exist yet."""
    sidecar = CORPUS / arm / "semantic_verdicts.jsonl"
    if not sidecar.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn final line from a kill; that candidate is rescored on resume
        out[record.pop("name")] = record
    return out


def audit_arm(arm: str, pooled_trials: int) -> dict[str, Any]:
    """Every layer's verdict for one arm, deflated at both the arm's and the pooled trial count."""
    paths = sorted((CORPUS / arm).glob("candidate_*.py"))
    backtests = json.loads(
        (CORPUS / arm / "backtest_results.json").read_text(encoding="utf-8"))

    n_trials = arm_trials(arm)
    static = static_layer(paths)
    per_arm, pbo = statistical_layer(backtests, n_trials)
    # Amendment 1.2(3): neither reading may ever be reported without the other. Computing both here
    # rather than at analysis time makes it awkward to publish one and forget the other.
    pooled, _ = statistical_layer(backtests, pooled_trials)
    semantic = semantic_so_far(arm)

    rejected = sum(1 for v in static.values() if v["rejected"])
    _log.info("%-3s n=%-4d trials=%-4d static rejected %-4d  semantic %d/%d scored  pbo=%s",
              arm, len(paths), n_trials, rejected, len(semantic), len(paths),
              "n/a" if pbo is None else f"{pbo:.3f}")

    return {
        "arm": arm, "paradigm": ARMS[arm], "n_candidates": len(paths),
        "n_trials_arm": n_trials, "n_trials_pooled": pooled_trials, "pbo": pbo,
        "semantic_coverage": len(semantic) / len(paths) if paths else 0.0,
        "static": static, "semantic": semantic,
        "statistical": per_arm, "statistical_pooled": pooled,
    }


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), default=None)
    args = parser.parse_args()

    pooled_trials = sum(arm_trials(a) for a in ARMS)
    _log.info("pooled trial count across all six arms: %d", pooled_trials)

    for arm in ([args.arm] if args.arm else list(ARMS)):
        payload = audit_arm(arm, pooled_trials)
        out = CORPUS / arm / "audit_results.json"
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        _log.info("written to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

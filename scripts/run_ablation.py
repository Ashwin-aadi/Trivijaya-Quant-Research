"""Ablate the three auditor layers and compare each combination against random rejection.

Seven combinations: each layer alone, each pair, and all three. For every one, candidates are
ranked by that combination's confidence that a strategy is *sound*, coverage is swept, and the
resulting curve is compared against a bootstrapped random-rejection baseline at matched coverage.

**The layers are combined here and nowhere else.** Upstream each layer's verdict is stored on its
own, so the ablation can form these combinations from raw signals rather than from a pre-merged
"flagged by any layer" column that would have destroyed the very comparison being made.

## The holdout gate

`AUAP` is defined on out-of-sample performance, and the holdout may be evaluated **once per
project** with the PI's explicit authorisation (Rule 7). This script therefore refuses to read the
holdout unless `--holdout` is passed together with `--authorised-by`, and it records that
authorisation in the run manifest. Without them it uses development-period performance and labels
the output a diagnostic, which is not the reportable AUAP.

The gate is structural on purpose. A convention that the holdout is only touched when intended is
worth very little at two in the morning near the end of a long run.

Usage:
    python scripts/run_ablation.py --corpus runs/<stamp>/candidates
    python scripts/run_ablation.py --corpus runs/<stamp>/candidates --holdout --authorised-by "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.log import get_logger  # noqa: E402
from src.eval.abstention import abstention_curve, beats_random, random_baseline  # noqa: E402

_log = get_logger(__name__)

LAYERS = ("static", "semantic", "statistical")

#: A Sharpe within this of zero means the strategy never took a position over the whole window.
FLAT_TOLERANCE = 1e-9


def soundness(audit: dict[str, Any], layer: str, name: str) -> float:
    """How strongly ``layer`` believes ``name`` is sound. Higher is kept first.

    Each layer reports confidence that something is *wrong*, so this is its negation. A candidate a
    layer never scored - it crashed, or the model errored - contributes 0.0: neither evidence for
    nor against, which keeps an unscored candidate from being ranked top merely by absence.
    """
    record = audit.get(layer, {}).get(name)
    if not record:
        return 0.0
    return -float(record.get("confidence", 0.0))


def combined(audit: dict[str, Any], layers: tuple[str, ...], names: list[str]) -> dict[str, float]:
    """Mean soundness across the layers in this combination."""
    return {
        name: sum(soundness(audit, layer, name) for layer in layers) / len(layers)
        for name in names
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--holdout", action="store_true",
                        help="evaluate on the holdout; requires --authorised-by")
    parser.add_argument("--authorised-by", type=str, default=None,
                        help="the PI's explicit authorisation for a holdout evaluation")
    args = parser.parse_args()

    if args.holdout and not args.authorised_by:
        _log.error(
            "the holdout may be evaluated once per project and only with explicit PI "
            "authorisation (Rule 7). Pass --authorised-by with the authorisation text, or omit "
            "--holdout to produce the development-period diagnostic."
        )
        return 2
    if args.authorised_by and not args.holdout:
        _log.error("--authorised-by given without --holdout; refusing to guess the intent")
        return 2

    audit_path = args.corpus.parent / "audit_results.json"
    backtest_path = args.corpus.parent / "backtest_results.json"
    if not audit_path.exists() or not backtest_path.exists():
        _log.error("need both %s and %s", audit_path, backtest_path)
        return 1

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    backtests = json.loads(backtest_path.read_text(encoding="utf-8"))

    # Which candidates are eligible to be ranked is decided on development data in both modes.
    # Deciding it on holdout flatness would define the population using the very out-of-sample
    # outcome being measured — a strategy that stopped trading in 2025 would be quietly removed
    # rather than scored as the zero it earned, and the surviving population would be selected on
    # the future. The population is fixed in sample; only the score comes from the holdout.
    eligible = {
        record["name"] for record in backtests
        if record["outcome"] == "evaluated" and record.get("sharpe") is not None
        and abs(float(record["sharpe"])) >= FLAT_TOLERANCE
    }

    if args.holdout:
        holdout_path = args.corpus.parent / "holdout_results.json"
        if not holdout_path.exists():
            _log.error("holdout results not found at %s", holdout_path)
            return 1
        backtests = json.loads(holdout_path.read_text(encoding="utf-8"))
        _log.warning("HOLDOUT EVALUATION, authorised by: %s", args.authorised_by)

    # Flat candidates — executed correctly over the full window and never took a position — are
    # excluded from the ranking. They are all tied at exactly 0.0, so they carry no ordering
    # information, and at 65% of the executed set they would dominate every retained set and drag
    # P(c) toward zero at every coverage regardless of what the auditor did. They remain counted in
    # the corpus statistics, where they are the central finding; not deleted, only unranked.
    # Eligibility is the development-period judgment above; the value is whatever this window says,
    # zeros included. A candidate that qualified in sample and then never traded in 2025 scores 0.0
    # and stays in, because that is its out-of-sample result.
    performance = {
        record["name"]: float(record["sharpe"])
        for record in backtests
        if record["outcome"] == "evaluated" and record.get("sharpe") is not None
        and record["name"] in eligible
    }
    if len(performance) < 10:
        _log.error("only %d candidates have a performance number; too few to sweep",
                   len(performance))
        return 1

    names = sorted(performance)
    _, _, auap_interval = random_baseline(performance, seed=42)

    rows: list[dict[str, Any]] = []
    for size in (1, 2, 3):
        for layers in combinations(LAYERS, size):
            curve = abstention_curve(combined(audit, layers, names), performance)
            rows.append({
                "layers": list(layers),
                "auap": curve.auap,
                "p_at_005": curve.performance[0],
                "p_at_100": curve.performance[-1],
                "beats_random": beats_random(curve, auap_interval),
                "curve": {"coverages": list(curve.coverages),
                          "performance": list(curve.performance)},
            })

    out = {
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "performance_source": "holdout" if args.holdout else "development",
        "reportable_auap": bool(args.holdout),
        "holdout_authorisation": args.authorised_by,
        "n_candidates": len(performance),
        "random_baseline_auap_interval": list(auap_interval),
        "combinations": rows,
    }
    suffix = "holdout" if args.holdout else "development"
    (args.corpus.parent / f"ablation_{suffix}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    label = "HOLDOUT (reportable)" if args.holdout else "development (DIAGNOSTIC ONLY)"
    print(f"performance source: {label}   n = {len(performance)}")
    print(f"random baseline AUAP 95% interval: "
          f"[{auap_interval[0]:.4f}, {auap_interval[1]:.4f}]\n")
    print(f"{'layers':<40} {'AUAP':>9} {'P(0.05)':>9} {'P(1.0)':>9}  beats random")
    for row in sorted(rows, key=lambda r: -r["auap"]):
        print(f"{'+'.join(row['layers']):<40} {row['auap']:>9.4f} {row['p_at_005']:>9.4f} "
              f"{row['p_at_100']:>9.4f}  {'yes' if row['beats_random'] else 'no'}")
    if not args.holdout:
        print("\nThese are development-period numbers. They are a diagnostic that the pipeline "
              "produces a curve, not the AUAP result, which is defined out of sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

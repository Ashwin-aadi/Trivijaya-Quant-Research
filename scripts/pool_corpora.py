"""Merge the two generation batches into one corpus view for the ablation.

The batches were generated separately — 300 candidates at indices 0-299, then 1250 at 300-1549 —
because the first was complete before the second was authorised. They are one experiment, not two:
the prompt digest, model tag, seed rule and backtest window are identical across both, and this
script asserts that rather than assuming it. If any of those differ the batches must not be pooled,
and pooling them anyway would silently mix two populations under one number.

Candidate names are unique across batches because the index is part of the filename, so the merged
maps cannot collide. That is checked too.

Usage:
    python scripts/pool_corpora.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.log import get_logger  # noqa: E402

_log = get_logger(__name__)

BATCHES: tuple[Path, ...] = (Path("runs/20260728T172115Z"), Path("runs/batch2"))
POOLED = Path("runs/pooled")

#: Fields that must agree across batches for pooling to be legitimate.
MUST_MATCH = ("model_tag", "prompt_digest", "base_seed")


def check_comparable(summaries: list[dict[str, Any]]) -> None:
    """Raise unless every batch was generated under identical conditions."""
    for field in MUST_MATCH:
        values = {json.dumps(s.get(field)) for s in summaries}
        if len(values) != 1:
            raise SystemExit(
                f"batches differ on {field}: {sorted(values)}. These are two experiments, not one; "
                "pooling them would report a mixture as a single population."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=POOLED)
    args = parser.parse_args()

    # One entry per batch. Summaries and audits are objects; backtests are lists of records.
    summaries: list[Any] = []
    backtests: list[Any] = []
    audits: list[Any] = []
    for batch in BATCHES:
        for name, sink in (
            ("generation_summary.json", summaries),
            ("backtest_results.json", backtests),
            ("audit_results.json", audits),
        ):
            path = batch / name
            if not path.exists():
                _log.error("%s missing; run the pipeline for that batch first", path)
                return 1
            sink.append(json.loads(path.read_text(encoding="utf-8")))

    # Holdout results are pooled only if every batch has them. A partial pool would quietly compute
    # AUAP over whichever batch happened to be evaluated, and report it as the whole corpus.
    holdout_paths = [b / "holdout_results.json" for b in BATCHES]
    holdouts = (
        [json.loads(p.read_text(encoding="utf-8")) for p in holdout_paths]
        if all(p.exists() for p in holdout_paths) else []
    )

    check_comparable(summaries)

    merged_backtests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for records in backtests:
        for record in records:
            if record["name"] in seen:
                _log.error("duplicate candidate name %s across batches", record["name"])
                return 1
            seen.add(record["name"])
            merged_backtests.append(record)

    merged_audit: dict[str, Any] = {
        "n_candidates": sum(a["n_candidates"] for a in audits),
        # The trial count is a property of the ledger, not of a batch, so the later value covers
        # every draw taken. Summing the batches would double-count the shared history.
        "n_trials": max(a["n_trials"] for a in audits),
        "pbo_per_batch": [a.get("pbo") for a in audits],
        "static": {}, "semantic": {}, "statistical": {},
    }
    for audit in audits:
        for layer in ("static", "semantic", "statistical"):
            merged_audit[layer].update(audit.get(layer, {}))

    if holdouts:
        merged_holdout: list[dict[str, Any]] = []
        for records in holdouts:
            merged_holdout.extend(records)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "candidates").mkdir(exist_ok=True)  # so --corpus resolves; sources stay in place
    (args.out / "backtest_results.json").write_text(
        json.dumps(merged_backtests, indent=2), encoding="utf-8"
    )
    (args.out / "audit_results.json").write_text(
        json.dumps(merged_audit, indent=2), encoding="utf-8"
    )

    if holdouts:
        (args.out / "holdout_results.json").write_text(
            json.dumps(merged_holdout, indent=2), encoding="utf-8"
        )

    print(f"pooled {len(BATCHES)} batches -> {args.out}")
    print(f"  backtests {len(merged_backtests)}")
    print(f"  holdout   {len(merged_holdout) if holdouts else 0}")
    for layer in ("static", "semantic", "statistical"):
        print(f"  {layer:<12} {len(merged_audit[layer])}")
    print(f"  trials    {merged_audit['n_trials']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Cross-check the Phase 2.0 burn-in index series against NSE's own published index bhavcopy.

The burn-in (`calendar_cnx100_burnin.parquet`) came from yfinance. Phase 1.0 established that
yfinance is not to be trusted without corroboration, so the PI required one independent check
(Checkpoint 2.0 question 5).

The source used here is NSE's static index bhavcopy archive (``ind_close_all_<DDMMYYYY>.csv``),
which is the exchange's own published closing index value. It is independent of yfinance in
provider, transport and file format. Two other routes were tried first and failed, and that is
recorded in the checkpoint rather than quietly omitted:

* ``jugaad_data.nse.index_df`` — NSE's JSON history endpoint returns non-JSON (blocked).
* ``niftyindices.com/Backpage.aspx`` — returns the HTML page rather than the JSON payload.

A sample rather than all 986 sessions: each session is one HTTP request against a public archive,
and a seeded sample of 60 spread across the window answers the question ("is this series the same
series NSE published?") without four figures of requests. The sample size is reported with every
number, as the charter requires.

Usage:
    python scripts/cross_check_burnin.py [--sample 60]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

import numpy as np
import polars as pl

from src.common.config import load_config
from src.common.io import read_parquet
from src.common.log import get_logger
from src.common.manifest import RunManifest
from src.stress.inputs import BURNIN_FILE

_log = get_logger(__name__)

#: Row labels NSE has used for this index in the index bhavcopy. BOTH are required: the index was
#: published as "CNX 100" until NSE renamed the CNX series to NIFTY in late 2015, so files from the
#: early burn-in carry the old name. Matching only the current name silently drops those sessions
#: from the comparison and understates coverage — which is exactly what the first run of this
#: script did, before the labels were checked against the actual archive.
NSE_INDEX_LABELS = frozenset({"nifty 100", "cnx 100"})


def _nse_close(day: date) -> float | None:
    """The NIFTY 100 closing value NSE published for ``day``, or None if unavailable."""
    from jugaad_data.nse import bhavcopy_index_raw

    try:
        text = bhavcopy_index_raw(day)
    except Exception as exc:  # noqa: BLE001 - a fetch failure is data, not a crash; counted below
        _log.warning("no index bhavcopy for %s: %s", day, exc)
        return None
    for line in text.splitlines():
        parts = line.split(",")
        if parts and parts[0].strip().lower() in NSE_INDEX_LABELS:
            return float(parts[5])
    # Also reached when the archive returns a 404 HTML page instead of a CSV, which happens for a
    # handful of dates. Counted as unavailable rather than treated as a mismatch.
    _log.warning("index bhavcopy for %s has no %s row", day, sorted(NSE_INDEX_LABELS))
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=60)
    args = parser.parse_args()

    cfg = load_config()
    path = cfg.paths.data_raw / BURNIN_FILE
    burnin = read_parquet(path).sort("session_date")

    rng = np.random.default_rng(cfg.meta.seed)
    n = min(args.sample, burnin.height)
    picks = sorted(rng.choice(burnin.height, size=n, replace=False).tolist())
    sampled = burnin[picks]

    tol = float(cfg.raw["data"]["prices"]["cross_check_rel_tol"])
    max_rate = float(cfg.raw["data"]["prices"]["max_discrepancy_rate"])

    with RunManifest(cfg, script="cross_check_burnin.py") as run:
        run.add_input(path)
        rows: list[dict[str, object]] = []
        for record in sampled.iter_rows(named=True):
            day, ours = record["session_date"], float(record["close"])
            theirs = _nse_close(day)
            if theirs is None:
                rows.append({"date": str(day), "ours": ours, "nse": None, "rel_diff": None})
                continue
            rel = abs(ours - theirs) / theirs
            rows.append({"date": str(day), "ours": ours, "nse": theirs, "rel_diff": rel})

        compared = [r for r in rows if r["rel_diff"] is not None]
        diffs = np.array([float(r["rel_diff"]) for r in compared]) if compared else np.array([])
        breaches = [r for r in compared if float(r["rel_diff"]) > tol]
        rate = len(breaches) / len(compared) if compared else None

        summary = {
            "source_ours": "yfinance:^CNX100",
            "source_theirs": (
                "NSE index bhavcopy (ind_close_all_<DDMMYYYY>.csv), "
                "'Nifty 100' / 'CNX 100' row"
            ),
            "sampled": n,
            "unavailable": len(rows) - len(compared),
            "compared": len(compared),
            "relative_tolerance": tol,
            "max_relative_difference": float(diffs.max()) if diffs.size else None,
            "median_relative_difference": float(np.median(diffs)) if diffs.size else None,
            "n_beyond_tolerance": len(breaches),
            "discrepancy_rate": rate,
            "max_acceptable_rate": max_rate,
            "verdict": (
                "unknown (nothing compared)"
                if rate is None
                else ("pass" if rate <= max_rate else "FAIL")
            ),
            "worst": sorted(
                compared, key=lambda r: -float(r["rel_diff"])  # type: ignore[arg-type]
            )[:5],
        }
        out = cfg.paths.data_processed / "burnin_cross_check.json"
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        run.note("discrepancy_rate", rate)
        run.note("compared", len(compared))

    print(json.dumps({k: v for k, v in summary.items() if k != "worst"}, indent=2, sort_keys=True))
    if summary["verdict"] == "FAIL":
        print("\nworst offenders:")
        print(pl.DataFrame(summary["worst"]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

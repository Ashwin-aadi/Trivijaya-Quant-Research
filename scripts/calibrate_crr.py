"""Calibrate counterfactual regime resampling on the real index series and report its fidelity.

Answers three questions with measurements rather than assertions:

1. What block length does the Politis & White (2004) rule select on actual NIFTY 100 returns?
2. Do synthetic paths at that block length reproduce the moments of the real series — and where do
   they fail? The charter requires the comparison to be reported, and failures are the point.
3. What does conditioning on the Phase 2.0 regime labels change?

Writes ``data/processed/crr_calibration.json``. No strategy is run here and no performance metric
is computed: this measures the resampler against the index, nothing else.

Usage:
    python scripts/calibrate_crr.py [--paths 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date

import numpy as np
import polars as pl

from src.common.config import load_config
from src.common.log import get_logger
from src.common.manifest import RunManifest
from src.stress.crr import (
    conditional_bootstrap_indices,
    optimal_block_length,
    stationary_bootstrap_indices,
)
from src.stress.inputs import load_index_closes
from src.stress.moments import compare_moments

_log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_config()
    dev_start = date.fromisoformat(str(cfg.raw["dates"]["dev_start"]))
    dev_end = date.fromisoformat(str(cfg.raw["dates"]["dev_end"]))

    series = load_index_closes(cfg).filter(
        (pl.col("session_date") >= dev_start) & (pl.col("session_date") <= dev_end)
    )
    closes = series["close"].to_numpy()
    returns = np.diff(closes) / closes[:-1]
    dates = series["session_date"].to_list()[1:]
    _log.info("%d development returns, %s .. %s", returns.shape[0], dates[0], dates[-1])

    with RunManifest(cfg, script="calibrate_crr.py") as run:
        estimate = optimal_block_length(returns)
        _log.info(
            "block length %.2f sessions (characteristic lag %d, bandwidth %d)",
            estimate.block_length,
            estimate.characteristic_lag,
            estimate.bandwidth,
        )

        started = time.perf_counter()
        unconditional = stationary_bootstrap_indices(
            returns.shape[0], estimate.block_length, args.paths, seed=cfg.meta.seed
        )
        draw_seconds = time.perf_counter() - started
        report = compare_moments(returns, unconditional)

        # Regime-conditional paths, using the Phase 2.0 labels aligned to the same sessions.
        labels_frame = pl.read_parquet(cfg.paths.data_processed / "regime_labels.parquet")
        aligned = (
            pl.DataFrame({"session_date": dates})
            .join(labels_frame.select("session_date", "state"), on="session_date", how="left")
        )
        conditional_report = None
        conditional_summary: dict[str, object] = {}
        if aligned["state"].null_count() == 0:
            labels = aligned["state"].to_numpy()
            conditional = conditional_bootstrap_indices(
                labels, estimate.block_length, args.paths, seed=cfg.meta.seed
            )
            conditional_report = compare_moments(returns, conditional)
            occupancy = {
                str(int(state)): int(count)
                for state, count in zip(*np.unique(labels, return_counts=True), strict=True)
            }
            conditional_summary = {"label_occupancy": occupancy}
        else:
            _log.warning(
                "%d sessions have no regime label; skipping the conditional comparison rather "
                "than silently dropping them",
                aligned["state"].null_count(),
            )

        payload = {
            "n_returns": int(returns.shape[0]),
            "window": {"start": str(dates[0]), "end": str(dates[-1])},
            "n_paths": args.paths,
            "block_length": {
                "sessions": estimate.block_length,
                "rule": "Politis & White (2004) automatic selection, stationary-bootstrap constant",
                "characteristic_lag": estimate.characteristic_lag,
                "bandwidth": estimate.bandwidth,
                "g_hat": estimate.g_hat,
                "d_hat": estimate.d_hat,
            },
            "draw_seconds_for_all_paths": draw_seconds,
            "unconditional": report.as_dict(),
            "conditional_on_regime": (
                {**conditional_summary, **conditional_report.as_dict()}
                if conditional_report is not None
                else None
            ),
        }
        out = cfg.paths.data_processed / "crr_calibration.json"
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        run.note("block_length", estimate.block_length)
        run.note("n_paths", args.paths)

    print(f"\nblock length: {estimate.block_length:.2f} sessions ({args.paths} paths "
          f"drawn in {draw_seconds:.2f}s)\n")
    header = (
        f"{'statistic':22s} {'real':>12s} {'synth mean':>12s} "
        f"{'2.5%':>12s} {'97.5%':>12s}  in?"
    )
    for label, rep in (("UNCONDITIONAL", report), ("CONDITIONAL", conditional_report)):
        if rep is None:
            continue
        print(f"--- {label} ---")
        print(header)
        for c in rep.comparisons:
            mark = "yes" if c.real_inside_interval else "NO"
            print(
                f"{c.name:22s} {c.real:12.5f} {c.synthetic_mean:12.5f} "
                f"{c.synthetic_p2_5:12.5f} {c.synthetic_p97_5:12.5f}  {mark}"
            )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

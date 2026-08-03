"""Measure whether daily bars can identify a market-impact model, before assuming they can.

Phase 3.0's halt condition asks for an honest answer to one question: is daily-bar data sufficient
for defensible impact estimation? The charter's own note anticipates that it may not be. That is a
prediction, not a measurement, and this script exists so the answer is the second thing.

Five measurements, each falsifiable, each reported whichever way it comes out:

D1  **Exponent.** Fit ``|return| = a * (relative traded value)^delta`` per symbol. The square-root
    law predicts ``delta = 0.5``. If the cross-sectional median lands near 0.5 with a tight spread,
    daily bars support the functional form; if symbols disagree wildly, they do not.

D2  **Transience.** Impact is a temporary price concession and reverses; information is permanent.
    Regress the following week's return on today's, restricted to the heaviest-volume sessions. A
    strongly negative slope means the volume-return relation is measuring liquidity demand. A slope
    near zero means it is measuring news, and calling it impact would be the unsupported assumption
    the PI forbade.

D3  **Extrapolation.** A daily bar's volume is of order one day's normal volume. A capacity study
    asks about orders of a fraction of a percent of it. Measure the distance in orders of magnitude
    between where the model is fitted and where it would be used.

D4  **Stability.** Split the window in half and refit. A parameter that does not survive its own
    sample split is not a parameter.

D5  **Amihud.** The control. ILLIQ is a measure rather than a model, so if D1-D4 are discouraging
    and D5 is stable, that locates precisely what daily bars can and cannot carry.

Writes ``data/processed/impact_identifiability.json`` and a run manifest. Reads nothing but the
development panel and the point-in-time universe; the holdout is not opened.

Usage:
    python scripts/diagnose_impact_identifiability.py
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from src.capacity.impact import (
    ElasticityFit,
    add_daily_measures,
    extrapolation_gap,
    fit_elasticity,
    minimum_detectable_beta,
    reversal_betas,
)
from src.common.config import Config, load_config
from src.common.io import read_parquet
from src.common.log import get_logger
from src.common.manifest import RunManifest

_log = get_logger(__name__)

#: A symbol needs this many usable sessions before its exponent is estimated at all. Two years of
#: trading: short enough to keep names that entered the universe late, long enough that a single
#: volatile quarter cannot set the slope.
MIN_SESSIONS = 500
#: Sessions for the reversal regression, which is restricted to a decile and so starts from less.
MIN_REVERSAL_SESSIONS = 60
#: One week. Long enough for a liquidity-demand concession to be repaid, short enough that ordinary
#: drift has not accumulated into the measurement.
REVERSAL_HORIZON = 5
#: The horizon sweep exists so D2 is not one number. One session is the **positive control**:
#: short-term reversal at a one-day horizon is a documented effect in equity markets generally and
#: is one of the factors P3 is scheduled to measure, so if this machinery cannot find a negative
#: beta there, the machinery is broken and its null at five days means nothing.
REVERSAL_HORIZONS = (1, 5, 10, 21)
#: The reversal test is run on the heaviest-volume tenth of sessions, where impact should be most
#: visible if it is visible at all. Also run unrestricted, as a contrast.
HEAVY_QUANTILE = 0.9


def _describe(values: list[float], name: str) -> dict[str, Any]:
    """Quantiles of a fitted-parameter distribution, always carrying the count that produced it."""
    array = np.asarray(values, dtype=float)
    return {
        "name": name,
        "n": int(array.size),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else float("nan"),
    }


def _universe_panel(cfg: Config) -> pl.DataFrame:
    """The development panel restricted to names the point-in-time universe ever held.

    Restricting to universe members rather than all 2,697 listed symbols keeps the diagnostic about
    the population a capacity study would actually trade. Membership is taken as "ever a member"
    rather than day-by-day because these are properties of a *symbol*, estimated over its whole
    history; a per-day restriction would chop each symbol's series at rebalance boundaries and
    estimate the exponent on fragments.
    """
    panel = read_parquet(cfg.paths.data_processed / "prices_adjusted.parquet")
    universe = read_parquet(cfg.paths.data_processed / "universe.parquet")
    members = universe["symbol"].unique().to_list()
    return panel.filter(
        pl.col("symbol").is_in(members)
        & (pl.col("session_date") >= cfg.dates.dev_start)
        & (pl.col("session_date") <= cfg.dates.dev_end)
    )


def _stability(frame: pl.DataFrame, fits: list[ElasticityFit]) -> dict[str, Any]:
    """D4: refit the exponent on each half of the window and compare, symbol by symbol."""
    dates = frame["session_date"].unique().sort()
    midpoint = dates[dates.len() // 2]
    halves = {
        "first": frame.filter(pl.col("session_date") < midpoint),
        "second": frame.filter(pl.col("session_date") >= midpoint),
    }
    per_half = {
        label: {f.symbol: f.delta for f in fit_elasticity(part, min_sessions=MIN_SESSIONS // 2)}
        for label, part in halves.items()
    }
    common = sorted(set(per_half["first"]) & set(per_half["second"]))
    first = np.array([per_half["first"][s] for s in common])
    second = np.array([per_half["second"][s] for s in common])
    spearman = stats.spearmanr(first, second) if len(common) > 2 else None
    return {
        "split_date": str(midpoint),
        "n_symbols_fitted_in_both_halves": len(common),
        "n_symbols_full_window": len(fits),
        "spearman_between_halves": None if spearman is None else float(spearman.statistic),
        "spearman_p_value": None if spearman is None else float(spearman.pvalue),
        "median_abs_difference": float(np.median(np.abs(first - second))) if common else None,
        "median_delta_first_half": float(np.median(first)) if common else None,
        "median_delta_second_half": float(np.median(second)) if common else None,
    }


def _amihud_stability(frame: pl.DataFrame) -> dict[str, Any]:
    """D5: is the Amihud measure itself stable across the sample split? The control for D4."""
    dates = frame["session_date"].unique().sort()
    midpoint = dates[dates.len() // 2]
    per_symbol = (
        frame.drop_nulls("amihud_term")
        .with_columns(half=pl.when(pl.col("session_date") < midpoint).then(0).otherwise(1))
        .group_by(["symbol", "half"])
        .agg(illiq=pl.col("amihud_term").mean(), n=pl.len())
        .filter(pl.col("n") >= MIN_SESSIONS // 2)
    )
    wide = per_symbol.pivot(on="half", index="symbol", values="illiq").drop_nulls()
    if wide.height < 3:
        return {"n_symbols": wide.height, "spearman_between_halves": None}
    spearman = stats.spearmanr(wide["0"].to_numpy(), wide["1"].to_numpy())
    return {
        "n_symbols": wide.height,
        "spearman_between_halves": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
        "median_illiq_first_half": float(np.median(wide["0"].to_numpy())),
        "median_illiq_second_half": float(np.median(wide["1"].to_numpy())),
    }


def main() -> int:
    cfg = load_config()
    with RunManifest(cfg, "diagnose_impact_identifiability") as run:
        run.add_input(cfg.paths.data_processed / "prices_adjusted.parquet")
        run.add_input(cfg.paths.data_processed / "universe.parquet")

        panel = _universe_panel(cfg)
        measured = add_daily_measures(panel, adv_window=cfg.constraints.adv_window_sessions)
        _log.info("panel: %d symbol-days, %d symbols", measured.height,
                  measured["symbol"].n_unique())

        # D1
        fits = fit_elasticity(measured, min_sessions=MIN_SESSIONS)
        deltas = _describe([f.delta for f in fits], "delta (square-root law predicts 0.5)")
        r2 = _describe([f.r_squared for f in fits], "per-symbol R^2 of the log-log fit")

        # D2, swept across horizons so the finding is a shape rather than a single coefficient.
        sweep: dict[str, dict[str, Any]] = {}
        for horizon in REVERSAL_HORIZONS:
            heavy_h = reversal_betas(
                measured, horizon=horizon, min_sessions=MIN_REVERSAL_SESSIONS,
                high_participation_quantile=HEAVY_QUANTILE,
            )
            all_h = reversal_betas(
                measured, horizon=horizon, min_sessions=MIN_SESSIONS,
                high_participation_quantile=None,
            )
            # The detectability bound is what makes the null a finding rather than a shrug.
            bound = minimum_detectable_beta(heavy_h)
            sweep[str(horizon)] = {
                "heavy_volume_decile": _describe(
                    [f.beta for f in heavy_h], f"reversal beta, heavy, h={horizon}"
                ),
                "all_sessions": _describe(
                    [f.beta for f in all_h], f"reversal beta, all, h={horizon}"
                ),
                "fraction_of_symbols_negative_heavy": float(
                    np.mean([f.beta < 0 for f in heavy_h])
                ),
                "detectability_heavy": {
                    "pooled_beta": bound.pooled_beta,
                    "pooled_standard_error": bound.pooled_standard_error,
                    "pooled_minimum_detectable_beta": bound.pooled_minimum_detectable_beta,
                    "median_symbol_minimum_detectable_beta": (
                        bound.median_minimum_detectable_beta
                    ),
                    "unweighted_mean_beta": bound.unweighted_mean_beta,
                    "unweighted_standard_error": bound.unweighted_standard_error,
                    "estimates_disagree_on_sign": bound.estimates_disagree,
                    "power": bound.power,
                    "alpha": bound.alpha,
                    "n_symbols": bound.n_symbols,
                },
            }

        # D3
        gap = extrapolation_gap(
            measured, target_participation=cfg.constraints.max_participation_rate
        )

        result: dict[str, Any] = {
            "n_symbol_days": measured.height,
            "n_symbols": measured["symbol"].n_unique(),
            "dev_start": str(cfg.dates.dev_start),
            "dev_end": str(cfg.dates.dev_end),
            "adv_window_sessions": cfg.constraints.adv_window_sessions,
            "min_sessions_per_symbol": MIN_SESSIONS,
            "d1_exponent": {"delta": deltas, "fit_r_squared": r2},
            "d2_transience": {
                "primary_horizon_sessions": REVERSAL_HORIZON,
                "positive_control_horizon_sessions": 1,
                "by_horizon": sweep,
            },
            "d3_extrapolation": {
                "target_participation": gap.target_participation,
                "observed_median_relative_value": gap.observed_median,
                "observed_p01_relative_value": gap.observed_p01,
                "orders_of_magnitude_below_data": gap.orders_of_magnitude,
                "fraction_of_sessions_below_target": gap.fraction_below_target,
                "n_symbol_days": gap.n_symbol_days,
            },
            "d4_stability": _stability(measured, fits),
            "d5_amihud": _amihud_stability(measured),
        }

        out = cfg.paths.data_processed / "impact_identifiability.json"
        out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        run.note("output", str(out))
        run.note("n_symbols_fitted", len(fits))
        _log.info("wrote %s", out)
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

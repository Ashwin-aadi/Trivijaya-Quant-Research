"""Cross-sectional quality composite for the NIFTY 100 universe.

Comparing a stock's momentum in raw percentage terms across very different eras -- the 2016
liquidity shock, the 2020 crash, the 2021 rally -- is misleading, because the typical dispersion
of returns changes over time. This module standardises each stock's momentum against its own
long-run mean and dispersion so that every period sits on a comparable footing before ranking.
"""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

_MOMENTUM_WINDOW = 252
_MAX_NAMES = 15


def _momentum_stats(panel: pl.DataFrame, window: int) -> pl.DataFrame:
    """Per-symbol mean and standard deviation of trailing momentum across the supplied panel."""
    ordered = panel.sort(["symbol", "session_date"])
    momentum = ordered.with_columns(
        (pl.col("adj_close") / pl.col("adj_close").shift(window).over("symbol") - 1.0).alias(
            "momentum"
        )
    ).drop_nulls("momentum")
    return momentum.group_by("symbol").agg(
        mean_momentum=pl.col("momentum").mean(),
        std_momentum=pl.col("momentum").std(),
    )


def _latest_row_per_symbol(frame: pl.DataFrame) -> pl.DataFrame:
    """The most recently dated visible row for each symbol."""
    ordered = frame.sort(["symbol", "session_date"])
    return ordered.group_by("symbol", maintain_order=True).last()


class QualityComposite(Strategy):
    """Standardises trailing momentum against each stock's own long-run mean and dispersion."""

    rationale = (
        "Raw momentum is not comparable across market regimes because the typical dispersion of "
        "returns shifts over time. Standardising each stock's trailing twelve-month return "
        "against its own long-run mean and volatility puts every period on the same footing, so "
        "the ranking reflects genuine relative strength rather than the regime the market "
        "happens to be in."
    )

    def __init__(self, panel: pl.DataFrame) -> None:
        # Computed once, across the full study panel, so every session's ranking is measured
        # against the same yardstick instead of recomputing noisy per-session statistics.
        self._stats = _momentum_stats(panel, _MOMENTUM_WINDOW)

    def generate(self, view: MarketView) -> Signal:
        frame = view.history(lookback=_MOMENTUM_WINDOW + 10).sort(["symbol", "session_date"])
        frame = frame.with_columns(
            (
                pl.col("adj_close") / pl.col("adj_close").shift(_MOMENTUM_WINDOW).over("symbol")
                - 1.0
            ).alias("momentum")
        )
        latest = _latest_row_per_symbol(frame).drop_nulls("momentum")
        latest = latest.join(self._stats, on="symbol", how="inner")
        latest = latest.with_columns(
            ((pl.col("momentum") - pl.col("mean_momentum")) / pl.col("std_momentum")).alias(
                "z_score"
            )
        )
        ranked = latest.sort("z_score", descending=True).head(_MAX_NAMES)
        names = ranked["symbol"].to_list()
        if not names:
            return Signal(information_available_at=view.as_of, weights={})
        weight = 1.0 / len(names)
        return Signal(information_available_at=view.as_of, weights={s: weight for s in names})

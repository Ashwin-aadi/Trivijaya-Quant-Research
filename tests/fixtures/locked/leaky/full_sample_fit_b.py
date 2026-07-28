"""Volatility-managed momentum overlay for the NIFTY 100 universe.

Momentum strategies suffer their worst drawdowns during volatility spikes, so scaling exposure
down in the highest-volatility episodes should improve risk-adjusted returns without giving up
much of the average-period upside. Classifying "unusually high" volatility requires a sense of
scale for this market, which this module estimates once from the complete run of the index's
realised volatility.
"""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

_VOL_WINDOW = 20
_MOMENTUM_WINDOW = 60
_HIGH_VOL_PERCENTILE = 0.90
_MAX_NAMES = 10
_LOW_VOL_SCALE = 1.0
_HIGH_VOL_SCALE = 0.25


def _index_realized_vol(panel: pl.DataFrame, window: int) -> pl.DataFrame:
    """Equal-weighted index proxy and its trailing realised volatility."""
    index_level = (
        panel.sort(["symbol", "session_date"])
        .group_by("session_date")
        .agg(pl.col("adj_close").mean().alias("index_level"))
        .sort("session_date")
    )
    return index_level.with_columns(
        pl.col("index_level").pct_change().rolling_std(window_size=window).alias("realized_vol")
    ).drop_nulls("realized_vol")


class VolManagedMomentum(Strategy):
    """Scales a simple momentum book down when the market's realised volatility is elevated."""

    rationale = (
        "Momentum crashes cluster in high-volatility episodes, so a book that trims exposure "
        "when volatility is unusually elevated should give up little upside while avoiding the "
        "worst drawdowns. What counts as unusually elevated is set once from the full history "
        "of the index's realised volatility, so the threshold is not chasing noise in any single "
        "session."
    )

    def __init__(self, panel: pl.DataFrame) -> None:
        vol_history = _index_realized_vol(panel, _VOL_WINDOW)
        # Fixed threshold, estimated once from every session the index has -- including
        # sessions that lie after any individual decision date the backtest will later evaluate.
        self._high_vol_cutoff = vol_history["realized_vol"].quantile(_HIGH_VOL_PERCENTILE)

    def _current_scale(self, view: MarketView) -> float:
        index_frame = _index_realized_vol(view.history(lookback=_VOL_WINDOW + 10), _VOL_WINDOW)
        if index_frame.is_empty():
            return _LOW_VOL_SCALE
        latest_vol = index_frame.sort("session_date")["realized_vol"][-1]
        if self._high_vol_cutoff is None or latest_vol is None:
            return _LOW_VOL_SCALE
        return _HIGH_VOL_SCALE if latest_vol >= self._high_vol_cutoff else _LOW_VOL_SCALE

    def generate(self, view: MarketView) -> Signal:
        frame = view.history(lookback=252).sort(["symbol", "session_date"])
        frame = frame.with_columns(
            (
                pl.col("adj_close") / pl.col("adj_close").shift(_MOMENTUM_WINDOW).over("symbol")
                - 1.0
            ).alias("momentum")
        )
        latest = frame.sort("session_date").group_by("symbol", maintain_order=True).last()
        latest = latest.drop_nulls("momentum")
        ranked = latest.sort("momentum", descending=True).head(_MAX_NAMES)
        names = ranked["symbol"].to_list()
        if not names:
            return Signal(information_available_at=view.as_of, weights={})
        scale = self._current_scale(view) / len(names)
        return Signal(information_available_at=view.as_of, weights={s: scale for s in names})

from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "This strategy identifies stocks experiencing significant price swings or range compression. "
        "High dispersion suggests increased potential for profit or loss, while low dispersion indicates less volatility but potentially profitable entry points."
    )

    def __init__(self, window_high: int = 20, window_low: int = 10, top_n: int = 10) -> None:
        self._window_high = window_high
        self._window_low = window_low
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_high + self._window_low)
        if history.height < self._window_high + self._window_low:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window_high)
        volume_history = view.history(lookback=self._window_high).select(
            pl.col("symbol").alias("symbol"), "volume"
        )
        high_low_range = (
            history.select(
                pl.col("symbol").alias("symbol"),
                (pl.col("high") - pl.col("low")).alias("range"),
            )
            .group_by("symbol")
            .agg(pl.col("range").mean().alias("avg_range"))
        )

        if high_low_range.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        volume_condition = (
            volume_history.group_by("symbol")
            .agg((pl.col("volume") / pl.col("volume").shift(1) - 1.0).alias("volume_change"))
            .sort("volume_change", descending=True)
            .head(self._top_n)
        )

        symbols = set(high_low_range.sort("avg_range", descending=True).select("symbol").to_series().to_list()) & set(
            volume_condition.select("symbol").to_series().to_list()
        )[: self._top_n]

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
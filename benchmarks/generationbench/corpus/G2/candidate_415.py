from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompressionStrategy(Strategy):
    rationale = (
        "Range compression occurs when a security's price fluctuates less than usual. "
        "This can signal that the market is consolidating and may be due for an breakout or reversal."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        min_highs = (
            history.group_by("symbol")
            .agg((pl.col("high").rolling_min(window_size=self._window).alias("min_high")))
            .select(["symbol", "min_high"])
        )

        max_lows = (
            history.group_by("symbol")
            .agg((pl.col("low").rolling_max(window_size=self._window).alias("max_low")))
            .select(["symbol", "max_low"])
        )

        range_diffs = min_highs.join(max_lows, on="symbol", how="inner").with_columns(
            (pl.col("min_high") - pl.col("max_low")).alias("range_diff")
        )

        if range_diffs.height < 2:
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = range_diffs.sort("range_diff", descending=True).select(
            "symbol"
        ).to_list()[0]

        top_5 = sorted_symbols[:5]
        weight = 1.0 / len(top_5)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_5},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
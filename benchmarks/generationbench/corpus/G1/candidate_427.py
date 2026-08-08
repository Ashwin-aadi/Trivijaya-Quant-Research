from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "During periods of low volatility (range compression), the market may be under pressure "
        "to break out. By identifying symbols with reduced price movement, we can target those "
        "with a higher likelihood of a breakout."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or len(history.columns) < 2:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily range
        ranges = (
            history[symbols]
            .select(
                pl.col("session_date"),
                (pl.col("high") - pl.col("low")).alias("range"),
            )
            .group_by("session_date")
            .agg(pl.col("range").mean().alias("avg_range"))
        )

        # Filter symbols with low average range
        low_range_symbols = (
            ranges.sort("avg_range", descending=False)
            .select(["session_date", "symbol"])
            .head(5)["symbol"]
        ).to_list()

        if not low_range_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Further filter by recent close to ensure it is within the window
        recent_closes = view.closes()
        filtered_symbols = [
            symbol for symbol in low_range_symbols if symbol in recent_closes.columns
        ]

        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in filtered_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
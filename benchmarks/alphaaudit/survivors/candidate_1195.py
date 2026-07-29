from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that the market is consolidating and may soon breakout. "
        "By identifying symbols with reduced price ranges over a lookback period, we can "
        "identify potential candidates for future movement."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate the range for each symbol
        ranges = (
            history.group_by("symbol")
            .agg(
                (pl.col("high") - pl.col("low")).alias("range"),
            )
            .sort("range", descending=False)
        )

        if ranges.height < 1:
            return Signal(information_available_at=stamp, weights={})

        # Identify symbols with the smallest range
        smallest_range_symbols = [str(ranges["symbol"].to_list()[0])]
        weight = 1.0 / len(smallest_range_symbols)

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in smallest_range_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
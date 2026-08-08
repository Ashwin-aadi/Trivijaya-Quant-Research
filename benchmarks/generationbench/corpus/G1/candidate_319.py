from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates increased market volatility and potential for price "
        "movement. By identifying symbols with reduced range over a recent period, we can "
        "identify those that might experience significant price movements in the near future."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the range for each symbol over the lookback period
        ranges = (
            history.groupby("symbol")
            .agg(
                pl.col("high").max().alias("high_max"),
                pl.col("low").min().alias("low_min"),
            )
            .with_columns(
                (pl.col("high_max") - pl.col("low_min")).alias("range")
            )
        )

        # Filter out symbols with too little range
        min_range = 0.5 * ranges["range"].mean()
        filtered_ranges = ranges.filter(pl.col("range") >= min_range)

        if filtered_ranges.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Identify the symbol with the smallest range as a candidate for future movement
        pick_symbol = filtered_ranges.sort("range").head(1)["symbol"].item()

        weight = 1.0
        return Signal(
            information_available_at=stamp,
            weights={pick_symbol: weight},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
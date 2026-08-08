from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the market is consolidating after a significant move. "
        "During such periods, price volatility decreases, often leading to a breakout in the near future. "
        "Trading during these periods can provide profitable opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the daily range for each symbol
        ranges = (
            history
            .select(pl.col("symbol"), (pl.col("high") - pl.col("low")).alias("range"))
            .group_by("symbol")
            .agg(pl.col("range").sum().alias("total_range"))
        )

        # Calculate the average range for each symbol over the window period
        avg_ranges = (
            history
            .select(pl.col("symbol"), (pl.col("high") - pl.col("low")).alias("range"))
            .group_by("symbol")
            .agg(
                (pl.col("range").sum() / self._window).alias("avg_range"),
                pl.col("adj_close").mean().alias("mean_price"),
            )
        )

        # Merge the total and average range data
        merged = ranges.join(avg_ranges, on="symbol", how="inner")

        # Identify symbols where the current range is significantly lower than the average
        compressed_symbols = (
            merged.filter(
                (merged["range"] / merged["avg_range"]) < 0.5
            )  # Adjust the threshold as needed
            .select(pl.col("symbol"))
            .to_series()
            .to_list()
        )

        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in compressed_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
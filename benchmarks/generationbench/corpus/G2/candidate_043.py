from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when the price movement between high and low "
        "is reduced significantly over a period. This suggests that market sentiment is "
        "becoming less volatile, potentially leading to an accumulation phase where prices "
        "could move in a specific direction. We can identify such periods by comparing daily "
        "ranges with their historical average."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        closes = view.closes(lookback=self._window)

        # Calculate the daily range for each symbol
        ranges = (
            history.select(
                pl.col("symbol"),
                (pl.col("high") - pl.col("low")).alias("range")
            )
            .group_by("symbol")
            .agg(pl.col("range").mean().alias("avg_range"))
        )

        # Calculate the compression factor for each symbol
        compressed_ranges = (
            history.with_columns(
                (pl.col("adj_close") - pl.col("adj_close").shift(1)).abs().alias("price_diff"),
                ((pl.col("high") - pl.col("low")) / ranges["avg_range"]).alias("compression")
            )
            .group_by("symbol")
            .agg(
                (pl.col("price_diff") > 0).sum().alias("trading_days"),
                (pl.col("compression") < 1).mean().alias("compression_factor")
            )
        )

        # Identify symbols with significant compression
        picks = compressed_ranges.select(pl.col("symbol")).filter(
            (pl.col("trading_days") >= self._window) & 
            (pl.col("compression_factor").mean() < 0.5)
        ).to_series().to_list()

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
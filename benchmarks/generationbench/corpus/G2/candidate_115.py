from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that a stock's volatility has decreased, suggesting "
        "that the market is losing interest or confidence. This can lead to a potential "
        "profit opportunity if we enter positions when the range starts to expand again."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily range for each symbol
        high_low_diff = (history.high - history.low).alias("range")
        avg_range = history.range.mean().alias("avg_range")

        # Normalize the ranges by their average to identify compression
        normalized_ranges = (
            history.select(
                pl.col("symbol"),
                (pl.col("high") - pl.col("low")) / avg_range * 100.0
            )
            .with_column(pl.lit(stamp).alias("session_date"))
            .sort("session_date", descending=True)
        )

        compressed_symbols = normalized_ranges.filter(
            (pl.col("range") <= 5) & (pl.col("range") >= -5)
        ).select(pl.col("symbol")).to_list()

        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in compressed_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
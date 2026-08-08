from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the market is consolidating and could be due for a "
        "breakout. By identifying symbols with reduced volatility, we can find potential candidates "
        "for future price movements."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the range for each symbol
        ranges = (
            history.groupby("symbol")
            .agg(
                (pl.col("high") - pl.col("low")).alias("range"),
                (pl.col("adj_close").last() / pl.col("adj_close").first() - 1).alias("return"),
            )
            .sort("range", descending=False)
        )

        # Filter symbols with range compression
        compressed_symbols = (
            ranges.filter(
                (ranges["range"] <= self._threshold) & (ranges["return"].abs() < 0.2)
            )["symbol"]
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
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest
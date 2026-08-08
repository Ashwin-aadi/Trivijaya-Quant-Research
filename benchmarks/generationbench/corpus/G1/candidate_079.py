from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when the price action within a bar is reduced. "
        "This suggests that the market might be preparing for significant movement in either direction."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the range for each symbol within the lookback period
        ranges = (
            history.group_by("symbol")
            .agg(
                (pl.col("high") - pl.col("low")).alias("range"),
                ((pl.col("close").last() / pl.col("open").first()) - 1.0).alias("return_ratio"),
            )
        )

        # Filter out symbols with insufficient data
        ranges = ranges.drop_nulls()

        # Identify symbols with significant range compression
        compressed_symbols = (
            ranges.filter((pl.col("range") < pl.col("range").mean() / 2))
            .select(pl.col("symbol"))
            .to_series()
            .to_list()
        )

        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Distribute the weight equally among the identified symbols
        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in compressed_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
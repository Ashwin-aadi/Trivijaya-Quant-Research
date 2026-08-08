from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when volatility decreases, causing price movements to be "
        "more subdued. This can indicate a lack of significant buying or selling pressure and may "
        "suggest an upcoming breakout. By identifying symbols with high range compression, we aim "
        "to capture the potential subsequent movement."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the range for each symbol
        ranges = (
            history.group_by("symbol")
            .agg(
                (pl.col("high") - pl.col("low")).alias("range"),
                (pl.col("adj_close").last() / pl.col("adj_close").first() - 1).alias("return_ratio"),
            )
            .with_columns((pl.col("range") / history["session_date"].diff().mean()).alias("std_dev_range"))
        )

        # Identify symbols with high range compression
        compressed_symbols = (
            ranges.filter(
                (pl.col("std_dev_range").lt(0.2)) &  # Adjust this threshold as needed
                (pl.col("return_ratio") < 0.1)       # Adjust this threshold as needed
            )
            .select("symbol")
            .unique()
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
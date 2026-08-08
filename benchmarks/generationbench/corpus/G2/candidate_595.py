from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression can indicate increased buying or selling pressure. "
        "When the range between high and low prices contracts significantly, it suggests "
        "that buyers are willing to accept a smaller profit margin or sellers are eager to sell at any price."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_range_ratio = {}
        for symbol in view.symbols:
            high_low_diff = float(history.select(pl.col("high").max().alias("max_high")).select("max_high").to_list()[0][0]) - \
                            float(history.select(pl.col("low").min().alias("min_low")).select("min_low").to_list()[0][0])
            range_ratio = high_low_diff / float(history.select(pl.col("close").mean().alias("avg_close")).select("avg_close").to_list()[0][0])
            symbol_range_ratio[symbol] = range_ratio

        sorted_ratios = sorted(symbol_range_ratio.items(), key=lambda x: x[1], reverse=True)
        if not sorted_ratios:
            return Signal(information_available_at=stamp, weights={})

        top_symbol = sorted_ratios[0][0]
        weight = 1.0
        return Signal(
            information_available_at=stamp,
            weights={top_symbol: weight},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
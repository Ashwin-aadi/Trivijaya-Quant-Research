from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when price volatility decreases, potentially leading to "
        "mean reversion. This can be identified by comparing the recent range with a longer-term "
        "range, where a smaller recent range relative to the long-term range suggests potential "
        "price action."
    )

    def __init__(self, short_window: int = 20, long_window: int = 60) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._long_window + self._short_window - 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        short_range = (
            history.groupby("symbol")
            .agg(
                pl.col("high").max().alias("recent_high"),
                pl.col("low").min().alias("recent_low"),
                (pl.col("adj_close").rolling_max(self._short_window) - 
                 pl.col("adj_close").rolling_min(self._short_window)).alias("short_range"),
            )
            .sort("symbol")
        )

        long_range = (
            history.groupby("symbol")
            .agg(
                pl.col("high").max().alias("long_high"),
                pl.col("low").min().alias("long_low"),
                (pl.col("adj_close").rolling_max(self._long_window) - 
                 pl.col("adj_close").rolling_min(self._long_window)).alias("long_range"),
            )
        )

        combined = short_range.join(long_range, on="symbol")
        compressed_symbols = (
            combined.filter(
                (combined["short_range"] / combined["long_range"]) < 0.5
            )["symbol"]
        ).to_list()

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
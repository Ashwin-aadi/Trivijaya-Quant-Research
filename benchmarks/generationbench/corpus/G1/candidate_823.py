from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression signals that the market is consolidating and may be setting up "
        "for a breakout. This strategy identifies symbols where the recent price range has "
        "compressed significantly relative to historical ranges."
    )

    def __init__(self, window: int = 20, threshold: float = 0.75) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = history["adj_close"].mean()
        std_dev = history["adj_close"].std()

        min_adj_close = history.select(pl.col("adj_close").min().alias("min"))
        max_adj_close = history.select(pl.col("adj_close").max().alias("max"))

        compression_ratio = (max_adj_close["max"] - min_adj_close["min"]) / (
            2 * std_dev
        )
        mean_compression_ratio = compression_ratio.mean()

        if mean_compression_ratio < self._threshold:
            symbols_with_low_range = history.select(
                pl.col("symbol"),
                compression_ratio,
            ).filter(compression_ratio <= mean_compression_ratio).select(pl.col("symbol"))

            picks: list[str] = [s for s in view.symbols if s in symbols_with_low_range.to_list()]
            if not picks:
                return Signal(information_available_at=stamp, weights={})

            weight = 1.0 / len(picks)
            return Signal(
                information_available_at=stamp,
                weights={symbol: weight for symbol in picks},
            )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest
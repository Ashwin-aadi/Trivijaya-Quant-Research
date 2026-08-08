from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that when price volatility decreases significantly over a "
        "short period, it often precedes an increase in future volatility. By identifying such "
        "periods, we can opportunistically enter positions before the expected rebound."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date").tail(self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily price range
        high_min = history.select(pl.col("high").min()).to_series().item()
        low_max = history.select(pl.col("low").max()).to_series().item()
        daily_range = (high_min - low_max).abs()

        # Compute mean and standard deviation of the ranges over the window
        mean_range = daily_range.mean()
        std_range = daily_range.std()

        # Identify days with compression below threshold
        compressed_days = daily_range.filter((daily_range < mean_range * self._threshold) & (daily_range > 0)).to_list()

        if not compressed_days:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if history.select(pl.col("adj_close").filter(pl.col("session_date").is_in(compressed_days))).height > 0]
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_series().item()
    assert isinstance(newest, date)
    return newest
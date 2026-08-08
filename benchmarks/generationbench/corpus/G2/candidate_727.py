from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when a stock's price volatility decreases significantly. "
        "During such periods, the market may experience reduced uncertainty and higher trading "
        "opportunities. Identifying stocks with high range compression can potentially lead to "
        "above-average returns."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the high-low range for each day
        history = (
            history.with_columns(
                (pl.col("high") - pl.col("low")).alias("range")
            )
            .sort("session_date", descending=False)
            .tail(self._window)
        )

        # Compute the mean and standard deviation of the ranges
        range_mean = history.select(pl.col("range").mean().alias("mean_range"))[0][0]
        range_std_dev = (
            history.with_columns((pl.col("range") - pl.col("range").mean()).pow(2))
            .select(pl.col("range").sum() / self._window)
            .select(((pl.col("range") ** 0.5).alias("std_range")))[0][0]
        )

        # Identify stocks with high range compression
        compressed_stocks = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            daily_ranges = [float(v) for v in history[symbol].to_list()]
            std_dev = sum([(x - range_mean) ** 2 for x in daily_ranges]) / self._window
            std_dev = std_dev**0.5
            if std_dev < 1.3 * range_std_dev:
                compressed_stocks.append(symbol)

        # Select the top N stocks based on range compression
        compressed_stocks = sorted(compressed_stocks, key=lambda s: -range_std_dev[symbol])
        weight = 1.0 / len(compressed_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in compressed_stocks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
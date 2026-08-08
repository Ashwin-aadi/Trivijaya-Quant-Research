from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when a stock's price fluctuates less within a given "
        "time frame. This suggests that the market may be digesting information or reducing "
        "uncertainty, potentially setting up for a breakout in either direction."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        range_compression_scores = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"]:
                continue
            daily_range = (
                (history.select(pl.col("high") - pl.col("low")) / history.select(pl.col("close").shift(1)))
                .with_columns(
                    ((pl.col("high") - pl.col("low")).mean() / pl.col("adj_close").std()).alias("range_ratio"),
                )
                .sort("session_date", descending=False)
            )

            if daily_range.height < self._window:
                continue

            recent_daily_range = daily_range.select(pl.col("range_ratio").tail(self._window))
            mean_recent_range = float(recent_daily_range["range_ratio"].mean())
            current_range_ratio = float(daily_range.filter(pl.col("session_date") == history["session_date"].max())["range_ratio"])
            range_compression_score = (current_range_ratio - mean_recent_range) / mean_recent_range

            if not pl.all(range_compression_score.is_nan()):
                range_compression_scores[symbol] = range_compression_score

        if not range_compression_scores:
            return Signal(information_available_at=stamp, weights={})

        top_symbol = max(range_compression_scores, key=range_compression_scores.get)
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
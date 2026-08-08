from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that the market is consolidating before a potential breakout. "
        "High dispersion in daily price ranges can indicate accumulation or distribution activity, "
        "potentially leading to a significant move."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        daily_ranges = (
            history.select(
                pl.col("symbol"),
                (pl.col("high") - pl.col("low")).alias("range")
            )
            .group_by("symbol")
            .agg(pl.col("range").mean().alias("avg_range"))
        )

        compressed_symbols = []
        for symbol in view.symbols:
            if symbol not in daily_ranges.columns:
                continue
            avg_range = float(daily_ranges[daily_ranges["symbol"] == symbol]["avg_range"])
            current_range = history[history["symbol"] == symbol]["high"].to_list()[-1] - history[history["symbol"] == symbol]["low"].to_list()[-1]
            if (current_range / avg_range) > self._threshold:
                compressed_symbols.append(symbol)

        weights = {s: 1.0 / len(compressed_symbols) for s in compressed_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the stock prices are consolidating within a narrow "
        "range. This can be an indication of upcoming breakout or trend reversal, making it a "
        "potentially profitable entry point."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily range
        high_low_range = (
            history.select(
                [
                    pl.col("symbol"),
                    (pl.col("high") - pl.col("low")).alias("range")
                ]
            )
            .sort("session_date", descending=True)
            .group_by("symbol")
            .agg(pl.max("range").alias("max_range"))
        )

        # Calculate average range over the window
        avg_range = high_low_range.select(
            (pl.col("max_range") / self._window).alias("avg_range")
        ).collect().to_dict(False)[0]["avg_range"]

        # Identify symbols with significant compression
        compressed_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            daily_ranges = [float(v) for v in history[symbol].select("range").to_list()]
            current_range = max(daily_ranges[-self._window:])
            if current_range / avg_range < self._threshold:
                compressed_symbols.append(symbol)

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
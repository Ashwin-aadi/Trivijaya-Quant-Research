from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when the high-low range of a security's price "
        "contracts over a period. This is often indicative of consolidation before "
        "a breakout or trend reversal. Securities with compressed ranges may be more "
        "likely to experience significant price movements in either direction."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols_with_range_compression = []

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            symbol_history = history[symbol]
            high_prices = [float(v) for v in symbol_history["high"].to_list()]
            low_prices = [float(v) for v in symbol_history["low"].to_list()]

            # Compute the daily range (high - low)
            ranges = [(h - l) for h, l in zip(high_prices, low_prices)]

            if len(ranges) < self._window:
                continue

            # Find the maximum and minimum range over the window
            max_range = max(ranges)
            min_range = min(ranges)

            # Compute the standard deviation of daily ranges
            std_dev = pl.Series(ranges).std()

            # Identify symbols with a significant decrease in their price range
            if max_range > 2 * std_dev and min_range < 0.5 * max_range:
                symbols_with_range_compression.append(symbol)

        weight = 1.0 / len(symbols_with_range_compression)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols_with_range_compression},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
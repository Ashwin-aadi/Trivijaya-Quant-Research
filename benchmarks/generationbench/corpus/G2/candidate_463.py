from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression is a phenomenon where the price range between high and low "
        "decreases over time. This can indicate that market participants are becoming more "
        "conservative in their trading behavior, potentially leading to a breakout in the near future."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width != len(view.symbols) + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the range compression for each symbol
        ranges_compression = []
        for symbol in view.symbols:
            highs = history[symbol]["high"].to_list()
            lows = history[symbol]["low"].to_list()

            if len(highs) < self._window or len(lows) < self._window:
                continue

            max_high = max(highs[-self._window:])
            min_low = min(lows[-self._window:])

            current_range = max_high - min_low
            previous_range = max(highs[:-1]) - min(lows[:-1])
            ranges_compression.append(
                (symbol, current_range / previous_range if previous_range != 0 else 0)
            )

        # Filter out symbols with no range compression
        filtered_ranges = [
            entry for entry in ranges_compression if entry[1] < 0.9 and entry[1] > 0.5
        ]

        # Sort by the degree of compression, from least to most compressed (i.e., highest potential breakout)
        sorted_ranges = sorted(filtered_ranges, key=lambda x: x[1], reverse=True)

        if not sorted_ranges:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [entry[0] for entry in sorted_ranges[:5]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
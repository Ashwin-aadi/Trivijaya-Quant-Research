from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that a period of reduced volatility is coming to an end, "
        "and prices are likely to move sharply. Entering positions before the break occurs can "
        "capitalize on this anticipated movement."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        ranges = []
        for symbol in symbols:
            highs = [float(v) for v in history[symbol].select("high").to_list()]
            lows = [float(v) for v in history[symbol].select("low").to_list()]
            daily_range = [h - l for h, l in zip(highs[1:], lows[:-1])]
            mean_range = sum(daily_range) / len(daily_range)
            ranges.append(mean_range)

        mean_range = sum(ranges) / len(ranges)
        compressed_symbols = [
            symbol
            for symbol, range_ in zip(symbols, ranges)
            if range_ < 0.9 * mean_range
        ]

        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in compressed_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
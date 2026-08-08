from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the price action is becoming less volatile and "
        "reversals are likely. This strategy identifies symbols where recent volatility has decreased significantly compared to their historical range."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = (history["adj_close"].mean()).round(2).to_list()[0]
        volatility = (
            ((history["high"] - history["low"]) / mean_close) * 100
        ).to_list()
        recent_volatility = [volatility[-i] for i in range(self._window, 0, -1)]

        if len(recent_volatility) < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_recent_volatility = sum(recent_volatility) / self._window
        compressed_symbols = [
            symbol
            for symbol in view.symbols
            if volatility[recent_volatility.index(min(recent_volatility))] <= 0.5 * mean_recent_volatility
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
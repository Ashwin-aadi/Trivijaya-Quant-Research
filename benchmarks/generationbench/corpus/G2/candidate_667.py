from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompressionStrategy(Strategy):
    rationale = (
        "Range compression indicates that a security has been trading in a narrower range "
        "than its historical average. This can happen when there is less volatility and market "
        "participants are more cautious or the market environment is generally tranquil. "
        "Securities with lower than normal price swings may be undervalued, presenting an "
        "opportunity for entry."
    )

    def __init__(self, lookback: int = 60, threshold: float = 0.5) -> None:
        self._lookback = lookback
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        mean_range = sum(high - low for high, low in zip(closes[1:], closes[:-1])) / (
            self._lookback - 1
        )

        recent_closes = view.closes(lookback=self._lookback)
        recent_ranges = [
            max(float(v), float(recent_closes[symbol].shift(-1).to_list()[-1]))
            - min(float(v), float(recent_closes[symbol].shift(-1).to_list()[-1]))
            for symbol in view.symbols
            if symbol in recent_closes.columns and len(recent_closes[symbol].drop_nulls().to_list()) >= self._lookback
        ]

        compressed_symbols = [symbol for symbol, r in zip(view.symbols, recent_ranges) if r / mean_range <= self._threshold]

        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in compressed_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
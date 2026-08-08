from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakout continuation strategy aims to capture the momentum after a breakout. "
        "After an upward breakout occurs, we hold positions in top-performing stocks over "
        "the next 30 days."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_weights = {s: 0 for s in view.symbols}
        top_symbols = sorted(
            [
                (symbol, float(closes[symbol].max()))
                for symbol in view.symbols
                if symbol in closes.columns and not closes[symbol].is_empty()
            ],
            key=lambda x: -x[1],
        )[:5]

        weight_distribution = [0.2, 0.18, 0.16, 0.14, 0.12]
        total_weight = sum(weight_distribution)

        for i, (symbol, _) in enumerate(top_symbols):
            symbol_weights[symbol] = weight_distribution[i] / total_weight

        return Signal(
            information_available_at=stamp,
            weights=symbol_weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
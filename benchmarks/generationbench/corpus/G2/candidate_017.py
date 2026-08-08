from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue after a strong price movement. By identifying symbols that "
        "have recently broken out and then continued their move, we can capture the momentum of "
        "such movements."
    )

    def __init__(self, breakout_window: int = 20, continuation_window: int = 10) -> None:
        self._breakout_window = breakout_window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._breakout_window + self._continuation_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = set()
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(prices) < self._breakout_window + self._continuation_window:
                continue

            breakout_price = max(prices[-self._breakout_window:])
            continuation_prices = prices[-(self._breakout_window + self._continuation_window) : -self._breakout_window]
            if all(price > breakout_price for price in continuation_prices):
                breakout_symbols.add(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
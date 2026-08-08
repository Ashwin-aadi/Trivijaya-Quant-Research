from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in the direction of their initial move. By identifying "
        "breakout symbols and tracking their subsequent performance, we can attempt to capture "
        "this continuation effect for potential profits."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(values) < self._window + 1:
                continue
            high = max(values[:self._window])
            low = min(values[:self._window])
            latest_close = values[-1]
            if (latest_close > high and low - high <= (high - low) / 2.0) or \
               (latest_close < low and high - low <= (high - low) / 2.0):
                breakout_symbols.append(symbol)

        weights = {}
        for symbol in breakout_symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(values) < self._window + 1:
                continue
            latest_close = values[-1]
            if (latest_close > max(values[:self._window]) and latest_close - max(values[:self._window]) >= (max(values[:self._window]) - min(values[:self._window])) / 2.0) or \
               (latest_close < min(values[:self._window]) and min(values[:self._window]) - latest_close >= (max(values[:self._window]) - min(values[:self._window])) / 2.0):
                weights[symbol] = 1.0 / len(breakout_symbols)

        if not weights:
            return Signal(information_available_at=stamp, weights={})
        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
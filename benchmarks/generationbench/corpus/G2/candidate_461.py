from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often lead to continuation of the trend. By identifying symbols that "
        "have recently broken out and then continued in their breakout direction for a "
        "certain period, we can capture this trend continuation effect."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            history_symbol = history[symbol]
            if len(history_symbol.to_list()) < self._window + self._lookback:
                continue

            close_changes = [
                float(v) / float(history_symbol.shift(i).to_list()[0]) - 1.0
                for i, v in enumerate(history_symbol.drop_nulls().to_list()[-self._window:])
            ]
            if all(c > 0 for c in close_changes):
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
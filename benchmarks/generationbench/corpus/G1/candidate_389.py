from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "After a breakout, the continuation pattern suggests that the stock will move in "
        "the same direction for a certain period. By identifying such patterns, we can "
        "potentially profit from the continuation of the trend."
    )

    def __init__(self, window: int = 20, continuation_window: int = 10) -> None:
        self._window = window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_window)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = set()
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(adj_closes) < self._window + 1:
                continue

            breakout = False
            for i in range(self._window - 1, len(adj_closes) - 1):
                if adj_closes[i] > max(adj_closes[:i]):
                    breakout = True
                    break

            if not breakout:
                continue

            for j in range(i + 1, min(len(adj_closes), i + self._continuation_window)):
                if adj_closes[j] < adj_closes[j - 1]:
                    breakout_symbols.add(symbol)
                    break

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
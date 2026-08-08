from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "After a strong breakout in one direction, there is often continued momentum. "
        "This strategy identifies stocks that have recently broken out and are still trending."
    )

    def __init__(self, window: int = 20, breakout_days: int = 5) -> None:
        self._window = window
        self._breakout_days = breakout_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._breakout_days)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].to_list()]
            if len(values) < self._window + 1:
                continue

            # Find the recent breakout point
            for i in range(self._window, len(values)):
                if (
                    (i == self._window or values[i] > max(values[:i]))
                    and i - self._window <= self._breakout_days
                ):
                    breakout_symbols.append(symbol)
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
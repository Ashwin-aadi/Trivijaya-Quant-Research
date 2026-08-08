from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in the direction of the breakout. "
        "This strategy identifies stocks that have recently broken out and "
        "continues to hold them for a short period, expecting the continuation."
    )

    def __init__(self, window: int = 20, continuation_window: int = 10) -> None:
        self._window = window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_window)
        if history.is_empty() or history.height < self._window + self._continuation_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(values) < self._window + self._continuation_window:
                continue

            # Check for breakout condition
            if (values[-1] > max(values[-self._window : -self._continuation_window])) or (
                values[-1] < min(values[-self._window : -self._continuation_window])
            ):
                breakout_symbols.append(symbol)

        # Filter symbols to ensure they have a clear continuation after the breakout
        continuation_symbols: list[str] = []
        for symbol in breakout_symbols:
            if (symbol not in history.columns) or (history[symbol].is_null().any()):
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(values) < self._window + self._continuation_window:
                continue

            # Check for continuation condition
            if (
                values[-1] > max(values[-self._window : -self._continuation_window])
                or values[-1] < min(values[-self._window : -self._continuation_window])
            ):
                break  # Break out of the inner loop to avoid redundant checks

            continuation_symbols.append(symbol)

        if not continuation_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(continuation_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in continuation_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
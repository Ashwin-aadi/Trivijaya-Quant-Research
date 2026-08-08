from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue after a significant close above or below the recent trading range. "
        "This strategy identifies stocks that have recently broken out and are still moving in the breakout direction."
    )

    def __init__(self, window: int = 20, continuation_window: int = 5) -> None:
        self._window = window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        breakout_sigs = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(close_values) < self._window + 1:
                continue

            # Find the breakout point
            for i in range(self._window, len(close_values)):
                if close_values[i] > max(close_values[:self._window]) or \
                   close_values[i] < min(close_values[:self._window]):
                    break_point = i - self._window
                    break

            if break_point == 0:
                continue  # No valid breakout found

            # Check continuation for at least the next `continuation_window` days
            if all(
                (close_values[i] > close_values[i-1] and
                 close_values[i] > close_values[break_point])
                or
                (close_values[i] < close_values[i-1] and
                 close_values[i] < close_values[break_point])
                for i in range(break_point + 1, break_point + self._continuation_window)
            ):
                breakout_sigs.append(symbol)

        if not breakout_sigs:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(breakout_sigs)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_sigs}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
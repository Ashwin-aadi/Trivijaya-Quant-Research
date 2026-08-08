from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in the direction of their initial move. Identifying stocks that "
        "have recently broken out and then continued to rise can provide a trading opportunity."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 5)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = set()
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_history = [float(v) for v in history[symbol].to_list()]
            if len(close_history) < self._window + 5:
                continue

            # Check for a break above the previous high or below the previous low
            breakout = False
            for i in range(self._window, self._window + 5):
                if close_history[i] > max(close_history[self._window - 1:i]):
                    breakout = "above"
                    break
                elif close_history[i] < min(close_history[self._window - 1:i]):
                    breakout = "below"
                    break

            # Check for continuation after the breakout period
            if breakout:
                continue_period = range(self._window + 5, self._window + 20)
                if any(
                    close_history[j] > close_history[j - 1]
                    if breakout == "above"
                    else close_history[j] < close_history[j - 1]
                    for j in continue_period
                ):
                    breakout_symbols.add(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
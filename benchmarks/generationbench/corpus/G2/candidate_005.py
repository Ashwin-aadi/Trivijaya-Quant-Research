from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in the direction of the breakout. By identifying symbols that "
        "have recently broken out and are still above their recent resistance levels, we can "
        "potentially profit from this continuation pattern."
    )

    def __init__(self, window: int = 20, breakout_window: int = 10) -> None:
        self._window = window
        self._breakout_window = breakout_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        breakout_symbols = []
        for symbol in symbols:
            if symbol not in history.column_names[1:]:
                continue

            close_history = [float(v) for v in history[symbol].to_list()[1:]]
            if len(close_history) < self._breakout_window + 1:
                continue

            breakout_price = max(close_history[-self._breakout_window:])
            last_close = close_history[-1]
            if last_close > breakout_price:
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbol_set = set(breakout_symbols)
        continuation_candidates = []
        for symbol in symbols:
            if symbol not in history.column_names[1:]:
                continue

            close_history = [float(v) for v in history[symbol].to_list()[1:]]
            last_close = close_history[-1]
            if len(close_history) < self._window + 1 or last_close <= max(close_history[-self._window:]):
                continue
            if symbol in breakout_symbol_set:
                continuation_candidates.append(symbol)

        top_n_symbols = continuation_candidates[:5]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
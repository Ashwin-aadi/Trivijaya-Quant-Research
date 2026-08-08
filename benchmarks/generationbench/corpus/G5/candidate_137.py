from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "After a significant breakout, the continuation pattern suggests that prices "
        "may move further in the direction of the initial breakout. This strategy "
        "identifies symbols that have recently broken out and are continuing to trend."
    )

    def __init__(self, window: int = 20, threshold_breakout: float = 0.1, threshold_continuation: float = 0.1) -> None:
        self._window = window
        self._threshold_breakout = threshold_breakout
        self._threshold_continuation = threshold_continuation

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            open_value = float(history[symbol][0])
            close_breakout = float(close_values[-1])

            if (close_breakout - open_value) / open_value >= self._threshold_breakout:
                breakout_symbols.append(symbol)

        continuation_symbols = []
        for symbol in breakout_symbols:
            history_window = view.history(lookback=2)
            if symbol not in history_window.columns:
                continue
            close_values_window = [float(v) for v in history_window[symbol].drop_nulls().to_list()]
            last_close = float(history_window[symbol][-1])
            second_last_close = float(history_window[symbol][-2])

            if (last_close - second_last_close) / second_last_close >= self._threshold_continuation:
                continuation_symbols.append(symbol)

        weights = {s: 1.0 / len(continuation_symbols) for s in continuation_symbols}
        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "After a strong breakout in one direction, the continuation of that trend is "
        "often observed. This strategy aims to identify such continuations by looking for "
        "a breakout followed by an immediate follow-through."
    )

    def __init__(self, lookback_window: int = 20) -> None:
        self._lookback_window = lookback_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_window + 1)
        if history.height < self._lookback_window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_series = [float(v) for v in history[symbol].to_list()]
            high_close_ratio = max(close_series[-2:] / close_series[:-2]) - 1.0
            if high_close_ratio > 0.05:  # Consider a relatively strict breakout condition
                breakout_symbols.append(symbol)

        continuation_symbols = []
        for symbol in breakout_symbols:
            if symbol not in history.columns:
                continue
            close_series = [float(v) for v in history[symbol].to_list()]
            high_close_ratio = max(close_series[-2:] / close_series[:-2]) - 1.0
            low_close_ratio = min(close_series[-2:] / close_series[:-2]) - 1.0
            if (high_close_ratio > 0.05 or low_close_ratio < -0.05):  # Follow-through condition
                continuation_symbols.append(symbol)

        weight = 1.0 / len(continuation_symbols) if continuation_symbols else 0.0
        return Signal(
            information_available_at=stamp, weights={s: weight for s in continuation_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
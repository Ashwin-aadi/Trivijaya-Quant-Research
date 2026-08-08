from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often lead to continuation patterns. If a stock breaks out of its recent "
        "range and continues in the breakout direction for several days, it may signal that "
        "the move is sustainable. This strategy looks for such continuation."
    )

    def __init__(self, window: int = 20, continuation_days: int = 3) -> None:
        self._window = window
        self._continuation_days = continuation_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].to_list()]
            high, low = max(values[-self._window:]), min(values[-self._window:])
            last_close = float(history[symbol][-1])
            if (last_close > high and last_close > history[symbol][-2]) or \
               (last_close < low and last_close < history[symbol][-2]):
                breakout_symbols.append(symbol)

        continuation_symbols: list[str] = []
        for symbol in breakout_symbols:
            forward_returns = [float(v) for v in view.closes(lookback=self._continuation_days)[symbol].to_list()]
            if any(r > 0.01 for r in forward_returns[-self._continuation_days:]):
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
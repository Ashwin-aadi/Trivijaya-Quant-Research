from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakout continuation strategies exploit the idea that after a stock breaks out of its "
        "recent range and experiences an initial sharp move, it may continue in that direction. "
        "This is based on the notion that breakout events often indicate underlying strength or "
        "weakness in the stock, which can persist."
    )

    def __init__(self, window: int = 20, breakout_window: int = 5) -> None:
        self._window = window
        self._breakout_window = breakout_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._breakout_window - 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        breakout_history = history.sort("session_date").tail(self._breakout_window)
        continuation_history = history.tail(self._window)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in breakout_history.columns or symbol not in continuation_history.columns:
                continue

            breakout_close = float(breakout_history[symbol]["adj_close"].to_list()[-1])
            breakout_high = max(float(high) for high in breakout_history[symbol]["high"].to_list())
            breakout_low = min(float(low) for low in breakout_history[symbol]["low"].to_list())

            continuation_close = float(continuation_history[symbol]["adj_close"].to_list()[0])
            if (breakout_high >= breakout_close and continuation_close > breakout_high) or \
               (breakout_low <= breakout_close and continuation_close < breakout_low):
                picks.append(symbol)

        picks = picks[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
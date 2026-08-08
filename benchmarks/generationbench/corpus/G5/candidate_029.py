from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakout continuation strategies look for stocks that have recently broken out of "
        "a range and continue to trend in the direction of the breakout. This strategy "
        "identifies such stocks by first detecting a valid breakout using strict criteria, then"
        "checking if the stock continues to move favorably over the subsequent period."
    )

    def __init__(self, window: int = 20, breakout_window: int = 5) -> None:
        self._window = window
        self._breakout_window = breakout_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._breakout_window)

        if history.height < self._window + self._breakout_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = set()
        for symbol in view.symbols:
            if symbol not in history.columns or symbol not in view.history().columns:
                continue
            recent_closes = [float(v) for v in view.closes()[symbol].to_list()]
            if len(recent_closes) < self._breakout_window + 1:
                continue

            # Check for a valid breakout during the _breakout_window period
            breakout_price = min(recent_closes[:self._breakout_window])
            breakout_found = False
            for i in range(self._breakout_window, len(recent_closes)):
                if recent_closes[i] > breakout_price:
                    breakout_found = True
                    break

            if not breakout_found:
                continue

            # Check that the stock continued to move favorably after the breakout
            continuation_found = False
            for i in range(self._breakout_window + 1, self._window + self._breakout_window):
                if recent_closes[i] < recent_closes[0]:
                    continuation_found = True
                    break

            if not continuation_found:
                continue

            breakout_symbols.add(symbol)

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
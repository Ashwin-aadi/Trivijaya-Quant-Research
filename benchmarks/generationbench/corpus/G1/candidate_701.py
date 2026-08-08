from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After an initial breakout, a continuation of the trend is often observed. This "
        "strategy looks for stocks that have recently broken out and continue to move in "
        "the direction of the breakout."
    )

    def __init__(self, window: int = 20, breakout_lookback: int = 10) -> None:
        self._window = window
        self._breakout_lookback = breakout_lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._breakout_lookback)

        if history.is_empty() or history.height < self._window + self._breakout_lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            breakout_condition = (
                (history["session_date"] >= stamp - date(self._breakout_lookback))
                & (history[symbol].shift(1).is_null())
                | (history[symbol] > history[symbol].shift(1))
            )
            if breakout_condition.any():
                continuation_condition = (
                    (history["session_date"] >= stamp)
                    & (history[symbol] > history[symbol].shift(1))
                )
                if continuation_condition.any():
                    breakout_symbols.append(symbol)

        weights = {s: 1.0 / len(breakout_symbols) for s in breakout_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Continuation breakout occurs when a stock that has broken out of its recent range "
        "continues to move in the direction of the breakout. This strategy captures such "
        "opportunities by identifying stocks that have recently broken out and are still "
        "moving in the same direction."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback)

        if history.height < self._window + self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = set()
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(close_values) < self._window + self._lookback:
                continue

            breakout = max(close_values[-self._window:])
            breakout_index = close_values.index(breakout)
            post_breakout = close_values[breakout_index:]

            if all(x > y for x, y in zip(post_breakout, close_values[breakout_index : self._lookback + breakoutr_index])):
                breakout_symbols.add(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in the direction of the breakout. By identifying "
        "breakout symbols and analyzing their continuation patterns, we can generate "
        "trading signals for potential follow-through."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback)

        if history.is_empty() or history.height < self._window + self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol]["adj_close"].drop_nulls().to_list()]
            if len(values) < self._window + self._lookback:
                continue

            breakout_price = max(values[-self._window:])
            continuation_price = min(values[-self._lookback:])

            if values[-1] == breakout_price and continuation_price > breakout_price:
                breakout_symbols.append(symbol)

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
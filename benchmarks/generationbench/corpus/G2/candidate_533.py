from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in the direction of the initial move. By identifying "
        "symbols that have recently broken out and are still above their previous support or "
        "resistance levels, we can predict a continuation of the trend."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback)
        if history.height < self._window + self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            recent_closes = [float(v) for v in history[history["symbol"] == symbol]["adj_close"].to_list()]
            if len(recent_closes) < self._window + self._lookback:
                continue

            breakout_price = max(recent_closes[-self._window:])
            breakout_date = recent_closes.index(breakout_price)

            for i in range(self._lookback):
                if recent_closes[breakout_date - 1 - i] < breakout_price:
                    break
            else:
                continue

            breakout_symbols.append(symbol)

        breakout_symbols = list(set(breakout_symbols))[: self._lookback]
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
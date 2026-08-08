from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "After a strong breakout, the continuation of that move is often reliable. This strategy "
        "identifies stocks that have shown strong upward or downward momentum and bets on their "
        "continuation."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            recent_close = float(view.latest_close()[symbol])
            breakout_price = max(values[:-1]) if recent_close > max(values[:-1]) else min(values[:-1])

            if recent_close > breakout_price + 0.02 * (max(values) - min(values)):
                breakout_symbols.append(symbol)
            elif recent_close < breakout_price - 0.02 * (max(values) - min(values)):
                breakout_symbols.append(symbol)

        breakout_symbols = list(set(breakout_symbols))
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
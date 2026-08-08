from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakout continuation strategies bet that stocks which have recently broken out of "
        "their ranges will continue in the breakout direction. This is based on the assumption "
        "that a significant price move indicates underlying strength or weakness."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            closes = [float(v) for v in history[symbol].to_list()]
            open_price = float(history[symbol][0])
            high_low_diff = max(closes[-self._window:]) - min(closes[-self._window:])
            breakout_condition = (
                (closes[-1] > max(closes)) if open_price < max(closes) else
                (closes[-1] < min(closes)) if open_price > min(closes) else False
            )
            if len(closes) >= self._window and high_low_diff > 0 and breakout_condition:
                breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[:self._top_n]
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts are often followed by continuation moves. Identifying a breakout and "
        "continuing to hold the position can provide profits if the breakout is genuine."
    )

    def __init__(self, window: int = 20, threshold: float = 0.03) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symb = None
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            last_close = float(closes[symbol].tail(1).to_list()[0])
            history = [float(v) for v in closes[symbol][:-1].drop_nulls().to_list()]
            if len(history) < self._window:
                continue
            max_high = max(history)
            if last_close >= max_high * (1 + self._threshold):
                breakout_symb = symbol
                break

        if not breakout_symb:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0
        return Signal(
            information_available_at=stamp,
            weights={breakout_symb: weight},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After identifying a strong breakout in the top-performing stocks over a short window, "
        "we look for continuation of this trend by holding these stocks for an extended period. "
        "This strategy aims to capture the momentum from the initial breakout."
    )

    def __init__(self, breakout_window: int = 10, hold_period: int = 30) -> None:
        self._breakout_window = breakout_window
        self._hold_period = hold_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._breakout_window + self._hold_period - 1)
        if closes.height < self._breakout_window + self._hold_period - 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symb: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._breakout_window + self._hold_period - 1:
                continue
            if values[-1] >= max(values[-self._breakout_window :]):
                breakout_symb.append(symbol)

        if not breakout_symb:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symb)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symb},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
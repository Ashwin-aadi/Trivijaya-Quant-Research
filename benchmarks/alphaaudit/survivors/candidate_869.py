from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After identifying a breakout in the short-term trend, continuation signals are "
        "generated based on further price movement. This strategy aims to capitalize on the "
        "momentum of the breakout."
    )

    def __init__(self, window: int = 20, follow_window: int = 10) -> None:
        self._window = window
        self._follow_window = follow_window

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
            if values[-1] >= max(values):
                breakout_symbols.append(symbol)

        continuation_weights: dict[str, float] = {}
        for symbol in breakout_symbols:
            follow_closes = view.closes(lookback=self._follow_window + 1)
            if symbol not in follow_closes.columns:
                continue
            follow_values = [float(v) for v in follow_closes[symbol].drop_nulls().to_list()]
            if len(follow_values) < self._follow_window + 1:
                continue
            if follow_values[-1] > max(follow_values[: self._follow_window]):
                weight = 1.0 / len(breakout_symbols)
                continuation_weights[symbol] = weight

        return Signal(
            information_available_at=stamp, weights=continuation_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
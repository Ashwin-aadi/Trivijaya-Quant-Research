from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "This strategy identifies price reversion opportunities by comparing daily closing "
        "prices against a trailing 50-day simple moving average (SMA). When prices deviate significantly from the SMA, it suggests a potential reversion, and we take action."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        sma = sum(closes[-self._window:]) / self._window
        deviations = [abs((c - sma) / sma) for c in closes]

        if max(deviations) > 0.1:
            top_symbol = view.symbols[0]
            weight = 1.0
            return Signal(
                information_available_at=stamp,
                weights={top_symbol: weight},
            )
        else:
            return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
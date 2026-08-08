from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price levels revert to the mean over time. By identifying stocks that have moved far "
        "away from their recent average price and betting on a reversion, we can capture "
        "momentum reversals."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_price = closes.mean().to_series()
        for symbol in view.symbols:
            if symbol not in mean_price.index:
                continue
            current_close = float(view.latest_close()[symbol])
            if (current_close - mean_price[symbol]) / mean_price[symbol] > 0.15:
                return Signal(
                    information_available_at=stamp,
                    weights={symbol: 1.0},
                )

        return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
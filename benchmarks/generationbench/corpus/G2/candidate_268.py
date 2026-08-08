from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Reversion to the mean suggests that after an asset's price deviates significantly from "
        "its historical average, it will tend to move back toward its long-term average. By buying"
        " assets with prices below their 20-day moving average and selling those above it, one can"
        " exploit this tendency for profit."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            latest_close = view.latest_close()[symbol]
            mean_price = (
                closes.select(pl.col(symbol))
                .sort("session_date")
                .tail(self._window)
                .select(pl.col(symbol).mean())
                .to_series()
                .to_list()[0]
            )
            if latest_close < mean_price:
                signals[symbol] = 1.0
            elif latest_close > mean_price:
                signals[symbol] = -1.0

        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
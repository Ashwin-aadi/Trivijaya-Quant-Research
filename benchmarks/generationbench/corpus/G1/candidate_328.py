from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reverts to the mean over time. By using a trailing average, we can "
        "identify stocks that have moved too far from their historical mean and "
        "are likely to revert back."
    )

    def __init__(self, window: int = 60, mean_window: int = 30) -> None:
        self._window = window
        self._mean_window = mean_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in closes.columns]
        prices = {symbol: float(closes[symbol].to_list()[-1]) for symbol in symbols}
        mean_prices = {
            symbol: float(closes[symbol].mean().item())
            for symbol in symbols
        }
        deviations = [
            (prices[symbol] - mean_prices[symbol]) / mean_prices[symbol]
            for symbol in symbols
        ]

        threshold = 0.15
        picks = [symbol for i, symbol in enumerate(symbols) if abs(deviations[i]) > threshold]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={p: weight for p in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest
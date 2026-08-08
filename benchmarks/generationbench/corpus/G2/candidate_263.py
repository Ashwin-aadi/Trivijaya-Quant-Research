from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "Dual momentum exploits two different time horizons: short-term price strength and "
        "long-term price direction. Strong short-term performance coupled with positive long-term "
        "trends suggests a high probability of continued outperformance."
    )

    def __init__(self, short_window: int = 20, long_window: int = 60) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._long_window + max(self._short_window, 1))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        short_returns = (closes / closes.shift(self._short_window) - 1.0).to_list()
        long_returns = (closes / closes.shift(self._long_window) - 1.0).to_list()

        short_tickers: list[str] = []
        for symbol in view.symbols:
            if f"{symbol}_short_return" not in locals():
                continue
            if max(short_returns) == float(local[f"{symbol}_short_return"][-1]):
                short_tickers.append(symbol)

        long_tickers: list[str] = []
        for symbol in view.symbols:
            if f"{symbol}_long_return" not in locals():
                continue
            if max(long_returns) == float(local[f"{symbol}_long_return"][-1]):
                long_tickers.append(symbol)

        short_tickers = short_tickers[: min(len(short_tickers), 5)]
        long_tickers = long_tickers[: min(len(long_tickers), 5)]

        if not short_tickers or not long_tickers:
            return Signal(information_available_at=stamp, weights={})

        combined_weights: dict[str, float] = {}
        for symbol in set(short_tickers + long_tickers):
            combined_weights[symbol] = 0.5

        return Signal(
            information_available_at=stamp,
            weights=combined_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
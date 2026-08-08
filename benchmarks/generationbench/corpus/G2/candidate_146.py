from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "High-volatility stocks are more likely to exhibit mean-reverting behavior in the short term. "
        "By scaling trends with their historical volatility, we can identify assets that have recently "
        "underperformed relative to their volatility and may revert towards their mean."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["close"]
        mean_close = closes.mean()
        std_dev_close = closes.std()

        # Calculate the z-score for each stock
        z_scores = (closes - mean_close) / std_dev_close

        # Filter symbols with negative z-scores to find underperformers relative to volatility
        underperforming_symbols = [symbol for symbol, z in zip(view.symbols, z_scores.to_list()) if z < 0]

        if not underperforming_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Allocate equal weight to all underperforming symbols
        weight = 1.0 / len(underperforming_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in underperforming_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
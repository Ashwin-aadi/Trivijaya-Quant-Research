from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend-following strategies seek to profit from trends while "
        "limiting risk by scaling position size based on historical volatility. High volatility "
        "indicates a more uncertain market environment where trends are less likely, so smaller "
        "positions should be taken. Conversely, low volatility suggests that trends are more "
        "likely and larger positions can be taken."
    )

    def __init__(self, window: int = 20, max_position_size: float = 1.0) -> None:
        self._window = window
        self._max_position_size = max_position_size

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns and volatility
        daily_returns = (history["adj_close"] / history["adj_close"].shift(1) - 1.0).alias("r")
        vol = daily_returns.abs().mean().item()

        # Determine the weight based on volatility
        if vol == 0:
            return Signal(information_available_at=stamp, weights={})

        top_symbol = history.sort("r", descending=True)["symbol"].first().item()
        weight = self._max_position_size / (1 + vol)

        return Signal(
            information_available_at=stamp,
            weights={top_symbol: weight}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
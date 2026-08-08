from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following seeks to capture trends while reducing risk by "
        "scaling investment based on historical volatility. High volatility periods suggest "
        "a greater chance of a trend forming, and thus we should increase our exposure. This "
        "strategy aims to benefit from trending markets without overexposure during volatile "
        "periods."
    )

    def __init__(self, window: int = 20, scaling_factor: float = 1.5) -> None:
        self._window = window
        self._scaling_factor = scaling_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            (history["adj_close"] / history["adj_close"].shift(1) - 1.0).alias("return")
        ).filter(pl.col("return").is_not_null())
        
        volatility = returns.std().item()
        mean_return = returns.mean().item()

        # Calculate the trend signal
        if (returns[history.height - 1] > mean_return and volatility < self._scaling_factor * mean_return) or \
           (returns[history.height - 1] < -mean_return and volatility > -self._scaling_factor * mean_return):
            target_weights = {s: 0.5 for s in view.symbols}
        else:
            target_weights = {}

        return Signal(
            information_available_at=stamp, weights=target_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
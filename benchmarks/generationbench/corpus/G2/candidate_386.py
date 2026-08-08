from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to capture trends by scaling trades based on "
        "volatility. High volatility periods are expected to be followed by higher returns due "
        "to increased momentum, and low volatility periods may indicate reduced market activity."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = [float(v) for v in view.closes().drop_nulls().to_list()]
        symbols = view.symbols

        # Calculate daily returns
        returns = [
            (c / c_prev - 1.0)
            if i > 0 and not pl.col("adj_close").is_null()[i]
            else 0.0
            for i, (_, c, _, _, _, _, _, _) in enumerate(history.iter_rows())
        ]

        # Calculate volatility over the window period
        volatility = (pl.DataFrame(returns).std() * 252 ** 0.5).to_list()[0]

        # Trend following signal: positive returns indicate a trend up
        trend = [1 if r > 0 else -1 for r in returns[-self._window :]]

        # Scale the weights based on volatility and trend
        weight_scaling_factor = (volatility + 1) * sum(trend)
        weights = {s: weight_scaling_factor / len(symbols) for s in symbols}

        return Signal(
            information_available_at=stamp,
            weights={k: float(v) for k, v in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
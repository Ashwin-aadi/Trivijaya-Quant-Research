from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend-following strategies exploit the tendency for assets to continue "
        "in their recent direction after a period of high volatility. High volatility can indicate "
        "that an asset is in a trending state, making it more likely that it will continue in its recent "
        "direction. By scaling trades based on historical volatility, one can capture gains while reducing risk."
    )

    def __init__(self, window: int = 20, trend_multiplier: float = 1.5) -> None:
        self._window = window
        self._trend_multiplier = trend_multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = history["adj_close"].mean().item()
        volatility = (
            (history["adj_close"] - mean_close).abs().mean().item() * self._trend_multiplier
        )

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            close_series = [float(v) for v in history[symbol].to_list()]
            if len(close_series) < self._window:
                continue

            trend_score = (close_series[-1] - close_series[0]) / volatility
            if trend_score > 0.5:
                signals[symbol] = 1.0 / len(signals)

        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().item()
    assert isinstance(newest, date)
    return newest
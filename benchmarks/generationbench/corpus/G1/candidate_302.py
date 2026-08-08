from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits the idea that high volatility periods are followed by "
        "reversion. By scaling trends with volatility, we can capture momentum while "
        "minimizing risk during volatile times."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        trends = (closes / closes.shift(1) - 1.0).cumsum()
        volatility = closes.abs().rolling_std(window=self._window)

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            trend = trends[symbol]
            vol = volatility[symbol]
            if len(trend.to_list()) < self._window or len(vol.to_list()) < self._window:
                continue

            latest_trend = trend[-1]
            latest_volatility = vol[-1]

            if abs(latest_trend) > (latest_volatility * self._threshold):
                signals[symbol] = 0.5
            else:
                signals[symbol] = -0.25

        total_weight = sum(signals.values())
        normalized_weights = {s: w / total_weight for s, w in signals.items() if w != 0}
        return Signal(
            information_available_at=stamp,
            weights={**normalized_weights, "CASH": 1 - sum(normalized_weights.values())},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
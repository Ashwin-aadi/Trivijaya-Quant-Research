from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits trends by scaling the portfolio allocation to symbols based on their volatility. "
        "High volatility indicates a stronger trend, leading to higher exposure."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility_scaled_weights: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].to_list()]
            mean_adj_close = sum(adj_closes) / len(adj_closes)
            returns = [(adj_closes[i] - adj_closes[i-1]) / adj_closes[i-1] if i > 0 else 0.0 for i in range(len(adj_closes))]
            mean_return = sum(returns) / len(returns)

            # Moving Average
            moving_average = history[symbol].rolling_mean(self._window).to_list()
            trend = (adj_closes[-1] - moving_average[-1]) / max(moving_average)
            volatility = (sum([(r - mean_return) ** 2 for r in returns]) / (len(returns) - 1)) ** 0.5

            if volatility == 0:
                continue
            weight = abs(mean_return) * trend / volatility * 3  # Scaling factor of 3
            volatility_scaled_weights[symbol] = weight

        total_weight = sum(volatility_scaled_weights.values())
        weights = {symbol: weight / total_weight for symbol, weight in volatility_scaled_weights.items()}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Trend following strategies leverage the historical volatility to scale "
        "positions in a way that reduces risk during volatile periods and allows for "
        "larger positions when markets are more stable."
    )

    def __init__(self, window: int = 20, scaling_factor: float = 1.0) -> None:
        self._window = window
        self._scaling_factor = scaling_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        volatility_scaled_trends = {}

        for symbol in symbols:
            adj_closes = history[symbol].to_list()
            returns = [(adj_closes[i] - adj_closes[i-1]) / adj_closes[i-1] for i in range(1, len(adj_closes))]
            mean_return = sum(returns) / len(returns)
            std_dev = (sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
            trend_strength = abs(mean_return / std_dev) * self._scaling_factor

            if view.latest_close()[symbol] >= max(adj_closes[-self._window:]):
                volatility_scaled_trends[symbol] = trend_strength

        if not volatility_scaled_trends:
            return Signal(information_available_at=stamp, weights={})

        total_trend_strength = sum(volatility_scaled_trends.values())
        weights = {s: v / total_trend_strength for s, v in volatility_scaled_trends.items()}
        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
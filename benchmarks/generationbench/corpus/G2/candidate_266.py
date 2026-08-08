from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "High volatility in a stock suggests that its price is likely to continue trending "
        "in the direction of its recent movement. By identifying such stocks, we can benefit "
        "from persistent trends without needing to predict their direction."
    )

    def __init__(self, window: int = 20, multiplier: float = 2.0) -> None:
        self._window = window
        self._multiplier = multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue

            # Calculate the rolling mean and standard deviation of returns
            returns = [(prices[i] - prices[i-1]) / max(prices[i-1], 1e-8) for i in range(1, len(prices))]
            rolling_mean_return = sum(returns) / self._window
            rolling_std_deviation = (sum([(r - rolling_mean_return)**2 for r in returns]) / self._window) ** 0.5

            # Check if the latest price is beyond a volatility threshold
            current_price = prices[-1]
            threshold = rolling_mean_return + self._multiplier * rolling_std_deviation
            if current_price > threshold:
                trends[symbol] = (current_price - threshold) / current_price

        # Select top symbols based on their deviation from the trend line
        picks: list[str] = sorted(trends, key=trends.get, reverse=True)[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
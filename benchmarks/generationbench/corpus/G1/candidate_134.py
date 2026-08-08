from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following involves identifying trends by measuring the "
        "volatility of asset prices. High volatility often suggests a stronger trend in one direction, "
        "and we can capitalize on this by buying or selling based on recent price movements and their volatility."
    )

    def __init__(self, window: int = 20, k: float = 1.0) -> None:
        self._window = window
        self._k = k

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        daily_returns = (closes[closes.columns[1:]] / closes[closes.columns[1:]].shift(1) - 1).drop_nulls()

        # Calculate rolling mean and standard deviation of returns
        rolling_mean = daily_returns.rolling_mean(window=self._window)
        rolling_std = daily_returns.rolling_std(window=self._window)

        # Define a threshold for identifying strong trends
        threshold = self._k * (rolling_std / rolling_mean).mean()

        # Identify symbols with significant trends
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in daily_returns.columns:
                continue
            returns = [float(v) for v in daily_returns[symbol].to_list()]
            mean_return = sum(returns[-self._window:]) / self._window
            if abs(mean_return) > threshold[symbol]:
                picks.append(symbol)

        # Determine the direction of the trend and allocate weights accordingly
        weight_per_pick = 1.0 / len(picks)
        weights = {s: weight_per_pick * (-1 if mean_return < 0 else 1) for s in picks}

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
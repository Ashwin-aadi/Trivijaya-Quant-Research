from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy follows a volatility-scaled trend of daily closing prices. "
        "It uses the 20-day standard deviation to scale the trend and identifies trends "
        "if the recent closing price is above or below the 50-day moving average of the scaled price."
    )

    def __init__(self, window_volatility: int = 20, window_trend: int = 50) -> None:
        self._window_volatility = window_volatility
        self._window_trend = window_trend

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_trend + 1)
        if closes.height < self._window_trend + 1:
            return Signal(information_available_at=stamp, weights={})

        trend_signals = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window_trend + 1:
                continue

            # Calculate daily returns
            returns = [(values[i] / values[i - 1]) - 1.0 for i in range(1, self._window_trend)]
            
            # Calculate volatility (20-day std of returns)
            vol = pl.Series(returns[-self._window_volatility:]).std()
            
            # Scale the last closing price by its volatility
            scaled_price = values[-1] / vol if vol > 0 else 1.0
            
            # Calculate 50-day moving average of scaled prices
            ma_scaled_prices = pl.Series(values[-self._window_trend:]).rolling_mean(self._window_trend).to_list()[-1]
            
            # Generate trend signal
            if values[-1] > ma_scaled_prices:
                trend_signals[symbol] = 1.0
            else:
                trend_signals[symbol] = -1.0

        # Select top 5 symbols based on trend signals
        picks = sorted(trend_signals, key=trend_signals.get, reverse=True)[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 0.20 / len(picks)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
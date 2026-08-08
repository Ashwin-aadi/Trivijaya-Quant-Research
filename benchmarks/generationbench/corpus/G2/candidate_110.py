from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to capture trends by adjusting the magnitude of "
        "trades based on recent volatility. Higher volatility suggests a stronger trend, which can "
        "lead to better risk-adjusted returns."
    )

    def __init__(self, window: int = 20, vol_window: int = 10) -> None:
        self._window = window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        recent_closes = history["adj_close"].to_list()
        recent_highs = history["high"].to_list()
        recent_lows = history["low"].to_list()

        if len(recent_closes) < self._window or len(recent_highs) < self._vol_window or len(recent_lows) < self._vol_window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        daily_returns = [(recent_closes[i] - recent_closes[i-1]) / recent_closes[i-1] for i in range(1, self._window)]
        
        # Calculate volatility using historical high and low prices
        vol = [abs(high - low) for high, low in zip(recent_highs[self._vol_window:], recent_lows[:self._window-self._vol_window])]
        avg_volatility = sum(vol) / len(vol)
        
        trend_score = 0.0
        if max(daily_returns) > 0:
            trend_score += (max(daily_returns) - min(daily_returns)) * 2
        
        # Scale the trend score by volatility
        scaled_trend = trend_score / avg_volatility

        top_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            latest_close = float(view.latest_close()[symbol])
            latest_high = max(float(high) for high in recent_highs)
            latest_low = min(float(low) for low in recent_lows)

            if scaled_trend > 0 and latest_close >= latest_high:
                top_symbols.append(symbol)
            elif scaled_trend < 0 and latest_close <= latest_low:
                top_symbols.append(symbol)

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
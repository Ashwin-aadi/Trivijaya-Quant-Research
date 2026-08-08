from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy follows the recent trend but scales exposure to volatility. "
        "High volatility periods reduce exposure while low volatility allows higher leverage."
    )

    def __init__(self, window: int = 20, vol_window: int = 10) -> None:
        self._window = window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._vol_window)

        if history.height < self._window + self._vol_window:
            return Signal(information_available_at=stamp, weights={})

        recent_closes = history["adj_close"].to_list()[-self._window:]
        recent_highs = history["high"].to_list()[-self._window:]
        recent_lows = history["low"].to_list()[-self._window:]

        trend_direction = 1 if recent_closes[0] < recent_closes[-1] else -1

        vol_series = [recent_highs[i] - recent_lows[i] for i in range(len(recent_highs))]
        volatility = sum(vol_series) / self._vol_window
        scaled_factor = max(0.5, 2 - (volatility * 2))  # Scale factor between 0.5 and 1.5

        symbol_weights: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_series = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(close_series) < self._window:
                continue

            trend_slope = (close_series[-1] - close_series[0]) / self._window
            weighted_trend_slope = trend_direction * trend_slope * scaled_factor
            symbol_weights[symbol] = max(0, min(1.0, abs(weighted_trend_slope)))

        return Signal(information_available_at=stamp, weights=symbol_weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
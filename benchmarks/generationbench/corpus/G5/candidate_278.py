from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy aims to follow trends by using a volatility scaling mechanism. "
        "High volatility periods suggest more cautious position sizing, while lower volatility"
        " indicates higher potential for trend continuation and thus larger positions."
    )

    def __init__(self, window: int = 20, scale_factor: float = 1.0) -> None:
        self._window = window
        self._scale_factor = scale_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        open_prices = history["open"]
        high_prices = history["high"]
        low_prices = history["low"]

        # Calculate daily true range
        tr = (pl.concat([high_prices - low_prices, high_prices - pl.col("close").shift(1), pl.col("close").shift(1) - low_prices]).max(axis=1)).to_list()

        close_values = [float(v) for v in closes.to_list()]
        open_values = [float(v) for v in open_prices.to_list()]

        # Calculate simple moving average of closing prices
        sma_close = float(sum(close_values) / self._window)
        # Calculate volatility using ATR (Average True Range)
        atr = sum(tr) / self._window

        if atr == 0:
            return Signal(information_available_at=stamp, weights={})

        # Trend direction based on close > open condition
        trend_direction = 1 if sma_close >= open_values[-1] else -1
        position_size = min(1.0, self._scale_factor * (sum(tr) / atr))

        selected_symbols = view.symbols
        weight_per_symbol = position_size / len(selected_symbols)

        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_symbol for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
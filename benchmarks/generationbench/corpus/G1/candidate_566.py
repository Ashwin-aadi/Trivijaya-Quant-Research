from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following involves entering positions when recent volatility "
        "is low relative to the current price level. This strategy aims to capture trending markets "
        "by using a simple moving average as a signal and adjusting for risk through volatility."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the rolling mean and standard deviation
        means = history.select(pl.col("adj_close").mean()).to_series().to_list()[0]
        stds = (history.select((pl.col("adj_close") - pl.lit(means).alias("diff")).std())
                .to_series().to_list()[0])

        # Calculate the trend signal
        signals = [float(v) for v in history["adj_close"].to_list()]
        mean_adj_close = means[-1]
        std_adj_close = stds[-1]

        if signals[-1] > (mean_adj_close + self._threshold * std_adj_close):
            direction = "up"
        elif signals[-1] < (mean_adj_close - self._threshold * std_adj_close):
            direction = "down"
        else:
            return Signal(information_available_at=stamp, weights={})

        # Determine the symbols to trade based on trend and volatility
        symbols_to_trade = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].unique().to_list():
                continue

            recent_closes = [float(v) for v in view.closes(lookback=self._window)[symbol].drop_nulls().to_list()]
            last_close = recent_closes[-1]
            mean_recent = sum(recent_closes) / len(recent_closes)

            if direction == "up" and last_close > (mean_recent + self._threshold * std_adj_close):
                symbols_to_trade.append(symbol)
            elif direction == "down" and last_close < (mean_recent - self._threshold * std_adj_close):
                symbols_to_trade.append(symbol)

        weights = {s: 1.0 / len(symbols_to_trade) for s in symbols_to_trade}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
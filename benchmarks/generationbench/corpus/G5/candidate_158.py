from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy identifies trends by measuring volatility and using moving averages. "
        "High volatility periods are indicative of potential trend changes, which can be used "
        "to enter or exit positions based on the relative position of closing prices to their moving average."
    )

    def __init__(self, window: int = 20, ma_window: int = 50) -> None:
        self._window = window
        self._ma_window = ma_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol = history["symbol"].to_list()[0]
        close_prices = [float(v) for v in history["close"].drop_nulls().to_list()]
        volatility = pl.Series(close_prices).rolling_std(window=self._window)
        ma_price = pl.Series(close_prices).rolling_mean(window=self._ma_window)

        if len(volatility.to_list()) < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_volatility = float(volatility.to_list()[-1])
        latest_ma_price = float(ma_price.to_list()[-1])

        # Determine trend direction based on volatility and relative close prices
        if (latest_volatility > pl.Series(close_prices).rolling_std(window=self._window - 5).mean()) and \
                close_prices[-1] > ma_price.to_list()[-2]:
            return Signal(
                information_available_at=stamp, weights={symbol: 0.95}
            )
        elif (latest_volatility < pl.Series(close_prices).rolling_std(window=self._window - 5).mean()) and \
                close_prices[-1] < ma_price.to_list()[-2]:
            return Signal(
                information_available_at=stamp, weights={symbol: -0.95}
            )
        else:
            return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
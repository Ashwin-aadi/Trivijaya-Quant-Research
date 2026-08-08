from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy identifies trends by scaling the daily price changes with their historical volatility. "
        "During upward trends, positions are taken to capitalize on the momentum."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_data = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            symbol_df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            close_prices = [float(v) for v in symbol_df["adj_close"].to_list()]
            daily_returns = [(close_prices[i] - close_prices[i-1]) / close_prices[i-1]
                             for i in range(1, len(close_prices))]
            volatility = (sum([r**2 for r in daily_returns])**0.5) * 252**(0.5/len(daily_returns))
            last_close_price = float(symbol_df.filter(pl.col("session_date") == symbol_df["session_date"].max())["adj_close"])
            if last_close_price >= max(close_prices):
                symbol_data[symbol] = volatility

        if not symbol_data:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbol_data)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbol_data.keys()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
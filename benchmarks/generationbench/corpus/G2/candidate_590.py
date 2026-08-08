from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySFTrendFollowing(Strategy):
    rationale = (
        "High volatility is often associated with mean reversion. By scaling the trend based "
        "on historical volatility, we can capture periods where prices are likely to revert "
        "to their mean after a significant move."
    )

    def __init__(self, window: int = 20, scaling_factor: float = 1.5) -> None:
        self._window = window
        self._scaling_factor = scaling_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.symbol.unique().to_list()]
        signals: dict[str, float] = {}

        for symbol in symbols:
            prices = history.select(pl.col("adj_close")).column(symbol).drop_nulls()
            log_returns = (prices / prices.shift(1) - 1.0).to_list()[1:]
            volatility = pl.DataFrame({"log_return": log_returns}).select(
                (pl.col("log_return").std() * self._scaling_factor).alias("volatility")
            ).height

            if volatility > 0:
                recent_close = view.latest_close()[symbol]
                last_price = history.select(pl.col("adj_close")).column(symbol)[-1]
                trend_signal = recent_close - last_price
                signals[symbol] = max(0, trend_signal / volatility)

        sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        if not sorted_signals:
            return Signal(information_available_at=stamp, weights={})

        top_symbol, _ = sorted_signals[0]
        weight = 1.0
        return Signal(
            information_available_at=stamp, weights={top_symbol: weight}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
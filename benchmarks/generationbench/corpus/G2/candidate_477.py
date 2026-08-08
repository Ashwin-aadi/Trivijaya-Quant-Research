from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to have higher returns than their high-volatility counterparts. "
        "This phenomenon is supported by academic research and can be exploited in a portfolio through "
        "tilting towards low-volatility assets."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the volatility (standard deviation of returns) for each stock
        volatilities: list[float] = []
        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        for symbol in symbols:
            close_prices = [
                float(v) for v in history[symbol].drop_nulls().to_list()
            ]
            if len(close_prices) < self._window:
                continue
            returns = [(close_prices[i] - close_prices[i-1]) / close_prices[i-1]
                       for i in range(1, len(close_prices))]
            volatility = (pl.Series(returns).std())  # Standard deviation of returns
            volatilities.append(volatility)

        # Find the N lowest volatilities
        low_vol_symbols = sorted(zip(symbols, volatilities), key=lambda x: x[1])[:5]

        if not low_vol_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(low_vol_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [symbol for symbol, _ in low_vol_symbols]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
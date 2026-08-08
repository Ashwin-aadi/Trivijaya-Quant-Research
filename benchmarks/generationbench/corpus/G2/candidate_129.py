from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "This is because low-volatility stocks are perceived by the market as less risky and "
        "therefore command higher valuations, leading to higher returns."
    )

    def __init__(self, window: int = 252) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        volatilities = {}
        for symbol in symbols:
            adj_close_series = history.select(pl.col("symbol") == symbol)["adj_close"]
            returns = (adj_close_series.shift(-1) / adj_close_series - 1.0).drop_nulls()
            volatility = returns.std().round(4)
            volatilities[symbol] = volatility

        sorted_symbols = sorted(volatilities, key=volatilities.get)
        top_n_low_volatility = sorted_symbols[:5]
        if not top_n_low_volatility:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_low_volatility)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_low_volatility},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max())["session_date"].to_list()[0]
    assert isinstance(newest, date)
    return newest
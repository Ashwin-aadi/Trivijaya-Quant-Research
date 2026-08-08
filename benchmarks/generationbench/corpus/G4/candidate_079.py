from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityDispersionStrategy(Strategy):
    rationale = (
        "This strategy exploits the behavior of stock prices during periods of high and low "
        "volatility. By identifying stocks with high price ranges (dispersion) when volatility is "
        "low, we aim to capture opportunities from increased market diversity."
    )

    def __init__(self, window: int = 20, dispersion_threshold: float = 5.0, volatility_threshold: float = 15.0) -> None:
        self._window = window
        self._dispersion_threshold = dispersion_threshold
        self._volatility_threshold = volatility_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        prices = history.select(
            pl.col("symbol"),
            (pl.col("high") - pl.col("low")).alias("price_range"),
            (pl.col("adj_close").shift(-1) / pl.col("adj_close") - 1).alias("daily_return"),
        )
        volatilities = (
            prices.group_by("symbol")
                  .agg((pl.col("daily_return").std().alias("volatility")))
                  .sort("volatility", descending=True)
        )
        dispersion_flag = (prices.select(pl.col("price_range")).max() - prices.select(pl.col("price_range")).min()) > self._dispersion_threshold
        volatility_flag = volatilities["volatility"] < self._volatility_threshold

        selected_symbols = set()
        for symbol in view.symbols:
            if dispersion_flag[0] and volatility_flag.get(symbol, False):
                selected_symbols.add(symbol)

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy exploits the low-volatility anomaly by overweighting stocks with lower historical volatility and underweighting those with higher volatility. The economic mechanism suggests that low-volatility stocks tend to outperform high-volatility stocks over time, driven by risk premiums."
    )

    def __init__(self, window: int = 250, min_stocks: int = 30) -> None:
        self._window = window
        self._min_stocks = min_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        daily_returns = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("daily_return")
            )
            .filter(pl.col("session_date") != view.as_of)
            .sort("session_date", descending=False)
        )

        volatilities = (
            daily_returns.groupby("symbol")
            .agg(
                (pl.col("daily_return").std().over(pl.col("session_date")).alias("volatility"))
            )
            .sort("volatility", descending=False)
        )

        if volatilities.height < self._min_stocks:
            return Signal(information_available_at=stamp, weights={})

        top_volatilities = [float(v[0]) for v in volatilities.head(self._min_stocks)["volatility"].to_list()]
        min_volatility = min(top_volatilities)

        weight_per_stock = 1.0 / self._min_stocks
        selected_symbols = {symbol: weight_per_stock if volatility == min_volatility else 0 for symbol, volatility in zip(volatilities["symbol"], top_volatilities)}

        return Signal(information_available_at=stamp, weights={k: v for k, v in selected_symbols.items() if v > 0})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
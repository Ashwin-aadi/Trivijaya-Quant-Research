from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy selects stocks with lower historical volatility to build a diversified "
        "portfolio aimed at reducing risk while potentially enhancing returns. Stocks are ranked based "
        "on their realized volatility over a 250-day rolling window and selected for inclusion if they "
        "fall below the median or average volatility."
    )

    def __init__(self, lookback: int = 250, max_portfolio_size: int = 50) -> None:
        self._lookback = lookback
        self._max_portfolio_size = max_portfolio_size

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        volatilities = (
            history.select(
                pl.col("symbol").alias("stock"),
                (pl.col("close") / pl.col("close").shift(1) - 1.0).rolling_std(window_size=self._lookback).alias("volatility"),
            )
            .group_by("stock")
            .agg(pl.col("volatility").mean().alias("avg_volatility"))
        )

        if volatilities.height < self._max_portfolio_size:
            return Signal(information_available_at=stamp, weights={})

        median_volatility = float(volatilities["avg_volatility"].median())
        low_vol_stocks = [stock for stock in view.symbols if float(volatilities.filter(pl.col("avg_volatility") <= median_volatility)["stock"].to_list())]

        if len(low_vol_stocks) > self._max_portfolio_size:
            low_vol_stocks = sorted(low_vol_stocks, key=lambda x: float(volatilities.filter(pl.col("stock") == x)["avg_volatility"].to_list()[0]), reverse=False)[:self._max_portfolio_size]

        weights = {s: 1.0 / len(low_vol_stocks) for s in low_vol_stocks}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
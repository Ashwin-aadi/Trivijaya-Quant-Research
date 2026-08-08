from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over time due to better risk management and more stable earnings. This strategy exploits this empirical phenomenon by weighting stocks based on their historical volatility."
    )

    def __init__(self, window: int = 252, num_stocks: int = 30) -> None:
        self._window = window
        self._num_stocks = num_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            closes.drop_nulls()
            .select(
                pl.col("session_date"),
                (pl.col(view.symbols[0]) / pl.col(view.symbols[0]).shift(1) - 1.0).alias("r")
            )
            .sort("session_date")
            .with_column(pl.col("r").rank(method="dense", descending=False))
        )

        # Calculate 1-year historical standard deviation of daily returns
        volatilities = (
            returns.group_by("symbol").agg(
                (pl.col("r") ** 2).mean().alias("variance"),
                ((pl.col("r") ** 2).mean() * self._window).alias("volatility")
            )
            .with_column((pl.col("volatility").rank(method="dense", descending=False)).alias("rank"))
        )

        # Select the lowest volatility quartile for long positions
        low_vol_stocks = volatilities.filter(pl.col("rank") <= (self._num_stocks // 4 + self._num_stocks % 4))
        low_vol_symbols = [s.strip() for s in low_vol_stocks["symbol"].to_list()]

        # Construct the signal
        weights = {s: 0.15 / len(low_vol_symbols) for s in low_vol_symbols}
        return Signal(
            information_available_at=stamp, 
            weights={**weights, **{s: -0.05 * (self._num_stocks // 4 + self._num_stocks % 4) / len(view.symbols) for s in view.symbols if s not in low_vol_symbols}}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
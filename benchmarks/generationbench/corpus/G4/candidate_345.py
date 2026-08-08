from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy exploits the empirical evidence that lower volatility stocks tend to outperform higher volatility stocks over long periods. By constructing a portfolio tilted towards low-volatility stocks, we aim to capture persistent outperformance while maintaining diversification and managing risk."
    )

    def __init__(self, lookback: int = 252, bottom_percentile: float = 0.5) -> None:
        self._lookback = lookback
        self._bottom_percentile = bottom_percentile

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .drop_nulls()
            .sort("session_date", descending=False)
        )

        # Compute historical volatility for each stock
        volatilities = (
            history.groupby("symbol")
            .agg(pl.sum("r").alias("returns"))
            .with_columns(
                (pl.col("returns") / self._lookback).std().alias("volatility")
            )
        )

        # Rank stocks by their historical volatility
        ranked_volatilities = volatilities.sort(by="volatility", descending=False)

        bottom_count = int(len(ranked_volatilities) * self._bottom_percentile)
        bottom_symbols = [s for s, _ in ranked_volatilities.head(bottom_count).iter_rows()]

        if not bottom_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(bottom_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in bottom_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
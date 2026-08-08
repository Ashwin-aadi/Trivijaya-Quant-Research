from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Trend following based on volatility scaling. During periods of high volatility, "
        "we expect more consistent trends and can enter positions with higher confidence."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            history.select([pl.col("symbol"), (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")])
            .drop_nulls()
            .sort("session_date")
        )

        # Calculate rolling volatility
        rolling_volatility = returns.groupby("symbol").agg(
            (pl.col("return").std().alias("volatility"))
        ).select(pl.col("symbol"), "volatility")

        # Filter symbols with high volatility
        high_volatility_symbols = (
            rolling_volatility.filter(
                pl.col("volatility") > self._threshold * returns.groupby("symbol").agg(pl.col("return").mean()).to_series()
            )["symbol"]
            .to_list()
        )

        if not high_volatility_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Allocate equal weight to each selected symbol
        weight = 1.0 / len(high_volatility_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                s: weight for s in high_volatility_symbols
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
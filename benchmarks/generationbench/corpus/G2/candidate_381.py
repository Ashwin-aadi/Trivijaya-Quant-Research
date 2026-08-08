from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "Low-volatility stocks tend to have higher returns over long periods. "
        "By tilting towards low-volatility stocks, we can aim for superior risk-adjusted returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        daily_returns = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .group_by("symbol")
            .agg(pl.col("r").mean().alias("avg_return"))
        )

        # Calculate historical volatility
        volatilities = (
            daily_returns.select(
                pl.col("symbol"),
                (pl.col("r") ** 2).sum() / self._window.alias("volatility_squared"),
                ((pl.col("r") * 100).round().cast(pl.Int64)).alias("rank")
            )
            .group_by("symbol", "avg_return")
            .agg(
                pl.col("volatility_squared").mean().alias("avg_volatility_squared"),
                pl.col("rank").min().alias("min_rank")
            )
        )

        # Filter out symbols with too little history
        volatilities = volatilities.filter(pl.col("avg_volatility_squared") > 0.0)

        if volatilities.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Rank by volatility and select the lowest ones
        sorted_symbols = (
            volatilities.sort("min_rank").select(["symbol"])
        ).to_dict(False)["symbol"]

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                s: weight for s in sorted_symbols[: self._window]
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
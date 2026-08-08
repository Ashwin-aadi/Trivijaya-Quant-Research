from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "Historically, low-volatility stocks tend to outperform high-volatility counterparts "
        "over long periods due to their lower risk and potentially higher stability during "
        "economic downturns. This strategy selects a portfolio of the least volatile stocks to "
        "capitalize on this anomaly."
    )

    def __init__(self, window: int = 252, num_stocks: int = 30) -> None:
        self._window = window
        self._num_stocks = num_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate log returns
        history = history.with_columns(
            (pl.col("close").rolling_window(pl.col("open") / pl.col("open").shift(1) - 1.0).mean()).alias("r")
        ).sort("session_date")

        # Compute volatility over the window
        volatilities = (
            history.drop(["symbol", "session_date"])
                .group_by("symbol")
                .agg((pl.col("r") ** 2).mean().alias("volatility"))
                .sort("volatility")
                .select("symbol", "volatility")
        )

        # Select the top N least volatile stocks
        picks = volatilities.head(self._num_stocks)["symbol"].to_list()

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
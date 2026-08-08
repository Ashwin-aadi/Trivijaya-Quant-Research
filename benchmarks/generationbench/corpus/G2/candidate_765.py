from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to have lower drawdowns and are less sensitive to market "
        "volatility. By tilting the portfolio towards low-volatility stocks, we aim to reduce overall "
        "portfolio risk while potentially increasing returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date")
            .drop_nulls()
        )

        # Calculate volatility for each stock over the lookback period
        volatilities = (
            history.groupby("symbol").agg(
                (pl.col("return").std().alias("volatility"))
            ).collect()
        )

        if volatilities.height == 0:
            return Signal(information_available_at=stamp, weights={})

        # Sort symbols by volatility and select top N
        sorted_symbols = volatilities.sort(pl.col("volatility"), descending=False)["symbol"].to_list()[:5]
        
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                s: weight for s in sorted_symbols
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
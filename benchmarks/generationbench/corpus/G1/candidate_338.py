from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy aims to follow the trend of high volatility stocks, "
        "as higher volatility often indicates greater potential for price movement."
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
        history_with_returns = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=False)
            .drop_nulls(subset=["symbol"])
        )

        # Filter out symbols with insufficient data
        if history_with_returns.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate volatility and rank symbols
        volatilities = (
            history_with_returns.group_by("symbol")
                .agg((pl.col("return").std().alias("volatility")))
                .sort("volatility", descending=True)
        )

        # Filter out symbols with too low volatility
        high_vol_symbols = volatilities.filter(pl.col("volatility") > self._threshold).select(["symbol"])

        if high_vol_symbols.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Allocate weight equally among selected symbols
        num_symbols = high_vol_symbols.height
        weight = 1.0 / num_symbols

        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in high_vol_symbols["symbol"].to_list()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
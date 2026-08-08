from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to have better risk-adjusted returns over the long term. "
        "This is due to their lower downside risk and can be exploited through tilting portfolios towards these stocks."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        )

        # Calculate rolling standard deviation as a proxy for volatility
        history = (
            history.with_columns(
                (pl.col("return").rolling_std(window=self._window, closed="both")).alias(
                    "volatility"
                )
            ).sort("session_date", descending=False)
        )

        # Get the most recent volatility values for each stock
        volatilities = history.select(
            pl.col("symbol"), pl.col("volatility").last().alias("volatility")
        ).collect()

        # Sort by lowest volatility and select top N symbols
        sorted_volatilities = (
            volatilities.sort("volatility", descending=False)
            .head(10)  # Adjust the number of top symbols to consider
            .select(pl.col("symbol"))
            .to_dict(True)
        )

        if not sorted_volatilities["symbol"]:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_volatilities["symbol"])
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol in sorted_volatilities["symbol"]
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
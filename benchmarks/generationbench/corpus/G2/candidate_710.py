from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility can be an indicator of future price movements. High volatility suggests "
        "that the market is uncertain and could experience a trend in either direction. By "
        "focusing on stocks with high recent volatility, we may identify those that are more "
        "likely to continue trending."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            (history["adj_close"] / history["adj_close"].shift(1) - 1.0)
            .with_columns(pl.col("adj_close").first().over(history["symbol"]).alias("prev_close"))
            .drop_nulls()
        )

        # Group by symbol and calculate the standard deviation of daily returns
        volatility = (
            returns.group_by("symbol")
                   .agg(
                       (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).mean().alias("avg_return"),
                       (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).std().alias("volatility")
                   )
        )

        # Sort by volatility in descending order
        ranked_volatility = volatility.sort("volatility", descending=True)

        if ranked_volatility.is_empty():
            return Signal(information_available_at=stamp, weights={})

        top_symbols = ranked_volatility.head(self._window)["symbol"].to_list()
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
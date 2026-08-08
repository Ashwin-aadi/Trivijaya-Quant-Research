from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion strategies exploit the tendency of asset prices to revert to their "
        "mean after deviating significantly. By identifying assets that have moved far from "
        "their recent mean price, we can make an argument for buying undervalued stocks and "
        "shorting overvalued ones."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the mean price over the lookback period
        mean_price = history.select(
            pl.col("symbol").alias("symbol"),
            (pl.col("adj_close") / 2).mean().alias("mean_price"),
        ).collect()

        # Merge with current closes to compute deviations from the mean
        merged_history = (
            history.lazy()
            .join(mean_price, on="symbol", how="left")
            .select(
                pl.col("session_date").alias("date"),
                pl.col("symbol").alias("symbol"),
                (pl.col("adj_close") - pl.col("mean_price")).alias("deviation"),
            )
        ).collect()

        # Filter for symbols with significant deviation
        filtered_history = merged_history.filter(
            (pl.col("deviation") > 2 * pl.col("mean_price").std()) | 
            (pl.col("deviation") < -2 * pl.col("mean_price").std())
        )

        if filtered_history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Compute weights for overvalued and undervalued symbols
        weight_overvalued = 1.0 / len(filtered_history.filter(pl.col("deviation") > 0).select(["symbol"]))
        weight_undervalued = -1.0 / len(filtered_history.filter(pl.col("deviation") < 0).select(["symbol"]))

        weights = {
            symbol: (weight_overvalued if deviation > 0 else weight_undervalued)
            for symbol, deviation in zip(
                filtered_history["symbol"].to_list(), 
                filtered_history["deviation"].to_list()
            )
        }

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
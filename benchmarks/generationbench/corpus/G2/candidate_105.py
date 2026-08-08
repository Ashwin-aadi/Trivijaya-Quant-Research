from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long run. "
        "This phenomenon can be attributed to reduced risk and higher certainty of returns."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the standard deviation of returns for each stock
        std_devs = (
            history.select(
                pl.col("symbol").alias("symbol"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
            )
            .group_by("symbol")
            .agg(pl.col("returns").std().alias("std_dev"))
        )

        # Sort by standard deviation and pick the lowest volatility stocks
        sorted_stocks = std_devs.sort("std_dev", descending=False)
        if sorted_stocks.height < 5:
            return Signal(information_available_at=stamp, weights={})

        top_stocks = [row["symbol"] for _, row in sorted_stocks.iter_rows().take(5)]
        weight = 1.0 / len(top_stocks)

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
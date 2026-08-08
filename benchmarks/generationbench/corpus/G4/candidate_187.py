from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DispersionStrategy(Strategy):
    rationale = (
        "This strategy exploits dispersion in the Indian equity market by identifying sectors or "
        "individual stocks with high price volatility. High dispersion periods are exploited to "
        "capture temporary inefficiencies through proportional allocation and periodic rebalancing."
    )

    def __init__(self, lookback: int = 30, top_n: int = 10) -> None:
        self._lookback = lookback
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty() or len(history["symbol"].unique()) < 5:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .filter(pl.col("session_date") >= date(2020, 1, 1))
            .sort("session_date", descending=False)
        )

        # Compute standard deviation of returns for each symbol
        dispersion = (
            history.group_by("symbol")
            .agg(
                pl.col("return").std().alias("dispersion"),
                pl.col("adj_close").last().alias("latest_close"),
            )
            .sort("dispersion", descending=True)
        )

        top_sectors = dispersion.head(self._top_n)["symbol"].to_list()
        weights = {s: 0.1 for s in top_sectors}

        return Signal(
            information_available_at=stamp, weights={k: v for k, v in weights.items() if v > 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
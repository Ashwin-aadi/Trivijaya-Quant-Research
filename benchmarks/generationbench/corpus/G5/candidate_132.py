from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have outperformed "
        "in the recent past to continue to outperform. This strategy aims to invest in the top "
        "performers based on their historical returns."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
            )
            .sort("session_date", descending=False)
        )

        # Compute rolling mean and standard deviation for returns
        grouped = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("returns").mean().alias("mean_return")),
                (pl.col("returns").std().alias("std_return")),
            )
        )

        if grouped.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate excess returns and rank by mean return
        history = (
            history
            .join(grouped, on="symbol")
            .with_columns(
                (pl.col("returns") - pl.col("mean_return")).alias("excess_returns"),
                (pl.col("excess_returns").rank(method="ordinal", descending=True)).alias("rank")
            )
        )

        # Get top N performers
        top_symbols = history.select(pl.col("symbol"), "rank").filter(
            (pl.col("session_date") == view.as_of) & (pl.col("rank") <= self._top_n)
        )["symbol"].to_list()

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
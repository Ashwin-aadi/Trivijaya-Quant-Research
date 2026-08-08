from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion(Strategy):
    rationale = (
        "Price reversion strategies exploit the tendency of prices to revert to their mean after "
        "deviating significantly from it. By identifying symbols that have deviated significantly "
        "from their historical price levels and betting against those deviations, we can capitalize "
        "on the reversal trend."
    )

    def __init__(self, window: int = 60, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        means = (
            history.groupby("symbol")
            .agg(
                (pl.col("adj_close").mean().alias("mean")),
                (pl.col("adj_close").std().alias("std")),
            )
            .select(["symbol", "mean", "std"])
        )

        latest_closes = view.closes(lookback=self._window)
        latest_prices = (
            means.join(latest_closes, on="symbol", how="inner")
            .with_columns(
                (pl.col("adj_close") - pl.col("mean")).alias("deviation"),
                ((pl.col("adj_close") - pl.col("mean")) / pl.col("std")).alias("z_score"),
            )
        )

        if latest_prices.is_empty():
            return Signal(information_available_at=stamp, weights={})

        thresholded = (
            latest_prices.filter(
                (pl.col("z_score").abs() > self._threshold)
            )
            .group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("reversion_signal"))
            .sort("reversion_signal", descending=True)
        )

        top_symbols = thresholded["symbol"].to_list()[:10]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
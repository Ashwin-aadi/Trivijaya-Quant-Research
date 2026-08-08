from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion occurs when a stock returns to its mean price level after deviating "
        "significantly. This strategy identifies stocks that have moved away from their 20-day"
        " average and are likely to revert."
    )

    def __init__(self, window: int = 20, std_dev_multiplier: float = 1.5) -> None:
        self._window = window
        self._std_dev_multiplier = std_dev_multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        means = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .with_column(
                (pl.col("adj_close") - pl.col("mean")).abs()
                .rank(method="dense", descending=True)
                .alias("reversion_rank")
            )
        )

        if means.is_empty():
            return Signal(information_available_at=stamp, weights={})

        filtered_history = history.join(means.select(["symbol", "reversion_rank"]), on="symbol")

        std_devs = (
            filtered_history.groupby("symbol")
            .agg(
                pl.col("adj_close").std().alias("std_dev"),
                (pl.col("adj_close") - pl.col("mean")).abs()
                / pl.col("std_dev")
                .rank(method="dense", descending=True)
                .alias("z_score")
            )
        )

        if std_devs.is_empty():
            return Signal(information_available_at=stamp, weights={})

        recent_z_scores = filtered_history.join(std_devs.select(["symbol", "z_score"]), on="symbol")

        top_symbols = (
            recent_z_scores
            .with_columns(
                (pl.col("z_score") > self._std_dev_multiplier).alias("outlier")
            )
            .filter(pl.col("outlier"))
            .select("symbol")
            .to_series()
            .to_list()[:5]
        )

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
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion is a phenomenon where asset prices and returns tend to move towards an "
        "average or mean over time. In the short term, extreme values are likely to revert back "
        "to their historical average, providing trading opportunities."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        means = closes.select(
            pl.col("symbol").alias("symbol"),
            (pl.col(pl.Series).mean().alias("mean")),
        ).collect()

        std_devs = closes.select(
            pl.col("symbol").alias("symbol"),
            (pl.col(pl.Series).std().alias("std")),
        ).collect()

        z_scores = (
            view.closes()
            .lazy()
            .group_by("symbol")
            .agg(
                ((pl.col("adj_close") - pl.col("mean")) / pl.col("std")).alias("z_score"),
            )
            .select(["symbol", "z_score"])
            .collect()
        )

        z_scores = (
            z_scores
            .with_columns(
                (pl.when(pl.col("z_score").abs() > 2.0).then(1.0)
                 .otherwise(0.0)).alias("outlier"))
            .group_by("symbol")
            .agg(
                pl.sum("outlier").alias("num_outliers"),
                pl.max("z_score").alias("max_z_score"),
                pl.min("z_score").alias("min_z_score"),
            )
        ).collect()

        picks: list[str] = []
        for symbol in view.symbols:
            if z_scores.height == 0 or z_scores[z_scores["symbol"] == symbol].is_empty():
                continue
            num_outliers = z_scores[z_scores["symbol"] == symbol]["num_outliers"].to_list()[0]
            max_z_score = z_scores[z_scores["symbol"] == symbol]["max_z_score"].to_list()[0]
            min_z_score = z_scores[z_scores["symbol"] == symbol]["min_z_score"].to_list()[0]

            if num_outliers > 0 and (max_z_score < -2.0 or min_z_score > 2.0):
                picks.append(symbol)

        weight = 1.0 / len(picks) if picks else 0.0
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
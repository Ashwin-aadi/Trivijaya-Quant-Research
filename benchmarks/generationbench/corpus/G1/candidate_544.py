from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class PriceLevelReversion(Strategy):
    rationale = (
        "Price reverts to the mean over time. Identifying price levels that have been "
        "extreme relative to their recent behavior can provide a basis for generating "
        "trading signals."
    )

    def __init__(self, window: int = 20, zscore_threshold: float = 1.5) -> None:
        self._window = window
        self._zscore_threshold = zscore_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        means = (
            closes.group_by("symbol")
            .agg((pl.col("adj_close").mean()).alias("mean"))
            .sort("symbol")
        )
        latest_closes = view.closes().select(["session_date", "symbol", "adj_close"])

        zscores = (
            latest_closes.join(
                means.select(["symbol", "mean"]),
                on="symbol",
                how="left",
            )
            .with_columns(
                (pl.col("adj_close") - pl.col("mean")).alias("diff"),
                ((pl.col("diff") / pl.col("mean").std().over("symbol"))).alias("zscore"),
            )
        )

        outliers = zscores.filter(pl.col("zscore").abs() > self._zscore_threshold)
        if outliers.height == 0:
            return Signal(information_available_at=stamp, weights={})

        symbols = [row["symbol"] for row in outliers.rows()]
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
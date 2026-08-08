from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Large volume moves in a single direction can indicate strong market sentiment. "
        "These moves often lead to sustained price movements and can be used for trend-following strategies."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily percentage change
        history = (
            history.with_column(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1).alias("return")
            )
            .with_column(
                (pl.col("volume") / pl.col("volume").shift(1)).alias("volume_ratio")
            )
            .sort("session_date", descending=False)
        )

        # Filter out symbols with insufficient history
        valid_symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if len(valid_symbols) < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the mean return and volume ratio
        mean_returns = (
            history[valid_symbols]
            .group_by("session_date")
            .agg(
                pl.col("return").mean().alias("avg_return"),
                pl.col("volume_ratio").mean().alias("avg_volume_ratio"),
            )
        )

        # Identify symbols with high volume and positive returns
        breakout_symbols = (
            mean_returns.filter(
                (pl.col("avg_return") > 0) & (pl.col("avg_volume_ratio") > 1.2)
            )
            .select(["session_date", "symbol"])
            .collect()["symbol"]
            .to_list()
        )

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Allocate equal weight to each symbol
        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
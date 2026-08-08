from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum involves buying stocks that have outperformed in the recent past. "
        "The idea is to capture the trend where winners tend to keep winning over short periods."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .drop_nulls(subset=["symbol", "session_date"])
            .sort("session_date")
        )

        # Compute cumulative returns
        history = (
            history.with_columns(
                (pl.col("r").cumsum()).alias("cum_r")
            )
            .filter(pl.col("cum_r") > 0.0)
            .group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("avg_close"), pl.col("cum_r").max().alias("max_cum_r"))
        )

        # Select top performing stocks
        sorted_history = history.sort("max_cum_r", descending=True)
        picks: list[str] = [row["symbol"] for row in sorted_history.to_dicts()[: self._top_n]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
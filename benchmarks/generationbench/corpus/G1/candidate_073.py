from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that security prices and valuations tend to drift back over time "
        "towards an average level. In a short horizon, stocks that have deviated significantly from "
        "their historical mean are expected to revert towards it."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .with_columns((pl.col("adj_close") - pl.col("mean")).abs().alias("deviation"))
            .sort("deviation", descending=True)
            .head(self._window)["symbol"]
        )
        
        if mean_close.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_n = mean_close.to_list()[: self._window]
        weight = 1.0 / len(top_n)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
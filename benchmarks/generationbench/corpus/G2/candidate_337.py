from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion suggests that prices which are far from their historical average "
        "tend to return towards the mean. By identifying such anomalies and taking positions "
        "in the direction of mean reversion, we can exploit this tendency for profit."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        means = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean().alias("mean")))
            .with_columns(
                (pl.col("adj_close") - pl.col("mean")).abs().alias("deviation")
            )
        )

        symbols_with_deviation = [
            symbol for symbol in view.symbols if symbol in means.columns
        ]
        if not symbols_with_deviation:
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_candidates = (
            means.sort("deviation", descending=True)
            .select(["symbol"])
            .head(self._window)[["symbol"]]
            .to_list()[0]
        )

        weight = 1.0 / len(mean_reversion_candidates)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in mean_reversion_candidates},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
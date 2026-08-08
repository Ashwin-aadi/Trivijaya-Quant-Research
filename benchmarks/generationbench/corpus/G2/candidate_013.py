from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion is a market phenomenon where prices that have deviated significantly "
        "from their mean level tend to revert back. Short-term deviations provide opportunities for profit."
    )

    def __init__(self, window: int = 10, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .with_columns((pl.col("adj_close") - pl.col("mean")).abs().alias("deviation"))
        )

        thresholded = (history
                       .join(mean_close, on="symbol", how="left")
                       .filter((pl.col("session_date") != view.as_of)
                               & (pl.col("deviation") > self._threshold * mean_close["mean"])))

        if thresholded.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = [row[0] for row in thresholded.sort("deviation", descending=True).select("symbol").to_series().to_list()]
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
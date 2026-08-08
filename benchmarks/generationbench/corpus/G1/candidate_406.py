from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Certain stocks tend to perform better at specific times of the year due to seasonal "
        "effects. By identifying these patterns, we can capitalize on them for trading."
    )

    def __init__(self, window: int = 365, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonality_data = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close") - pl.col("adj_close").shift(365)) / 365.0
            )
            .group_by("symbol")
            .agg((pl.col("adj_close") - pl.col("adj_close").shift(365)) / 365.0)
        )

        strong_seasonals = (
            seasonality_data.filter(
                (pl.col("adj_close") / 365.0).abs() > self._threshold
            )
            .sort("adj_close", descending=True)
            .select(pl.col("symbol"))
            .to_numpy()[0]
        )

        if not strong_seasonals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(strong_seasonals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in strong_seasonals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
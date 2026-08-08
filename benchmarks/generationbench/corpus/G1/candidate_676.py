from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion strategies assume that a security will return to its mean price level. "
        "By using the trailing 20-day average, we can identify instances where the current price is far from this mean, "
        "indicating potential for mean reversion."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean().alias("mean")))
            .with_columns(pl.col("mean") / pl.col("adj_close").shift(self._window - 1) - 1.0)
            .filter(pl.col("mean") < -0.25)
        )

        if mean_close.height == 0:
            return Signal(information_available_at=stamp, weights={})

        symbols = mean_close.select("symbol").to_list()[0]
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
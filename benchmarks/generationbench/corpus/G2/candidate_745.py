from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion strategies exploit the tendency of security prices to revert "
        "to their historical mean. In a short horizon, extreme deviations from the mean "
        "are likely to reverse over time."
    )

    def __init__(self, window: int = 5, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the mean of the last 'window' days for each symbol
        means = (
            closes
            .group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .with_columns((pl.col("adj_close") - pl.col("mean")).abs().alias("deviation"))
        )

        # Determine symbols with high deviations from the mean
        symbols = means.filter(
            (pl.col("deviation") / pl.col("mean") > self._threshold)
            & (pl.col("session_date").is_in(closes["session_date"]))
        )["symbol"].to_list()

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the mean reversion signal
        weights = {s: 1.0 / len(symbols) for s in symbols}
        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
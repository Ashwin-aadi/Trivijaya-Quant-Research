from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "Reversion to the mean is a common trading theme. If prices have deviated significantly "
        "from their historical average, they often tend to revert back towards that average. "
        "This strategy aims to exploit such deviations."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            closes.drop_nulls()
            .group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .with_columns((pl.col("adj_close") - pl.col("mean")).abs().alias("deviation"))
        )

        symbols_with_high_deviation = mean_close.sort("deviation", descending=True)["symbol"].to_list()[: self._window]

        if not symbols_with_high_deviation:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_with_high_deviation)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols_with_high_deviation},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
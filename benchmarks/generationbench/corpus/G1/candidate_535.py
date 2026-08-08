from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that the market is consolidating and may soon break out "
        "in one direction. By identifying symbols with reduced price ranges over a recent period, "
        "we can identify potential breakout candidates."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        range_compression = (
            history.select(
                pl.col("symbol"),
                (pl.col("high") - pl.col("low")).alias("range_diff")
            )
            .group_by("symbol")
            .agg(pl.col("range_diff").min().alias("min_range"))
            .sort("min_range", descending=False)
            .select("symbol")
        )

        symbols = range_compression["symbol"].to_list()[:5]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max())["session_date"][0]
    assert isinstance(newest, date)
    return newest
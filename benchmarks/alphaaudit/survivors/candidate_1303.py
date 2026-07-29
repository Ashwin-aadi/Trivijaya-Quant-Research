from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that the market is consolidating, and may be setting up "
        "for a breakout. By identifying symbols where the range has compressed significantly, "
        "we can capture potential moves before they occur."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        range_compression_df = (
            history
            .select(
                pl.col("symbol"),
                (pl.col("high") - pl.col("low")).alias("range")
            )
            .group_by("symbol")
            .agg(
                ((pl.col("range").max() / pl.col("range").mean()) * 100).alias("compression_ratio")
            )
            .sort("compression_ratio", descending=True)
        )

        if range_compression_df.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in range_compression_df.to_dicts()[:5]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
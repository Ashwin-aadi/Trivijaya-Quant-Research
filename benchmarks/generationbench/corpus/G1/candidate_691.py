from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression signals that the market is consolidating and could be setting up "
        "for a breakout or reversal. This strategy identifies symbols where the daily price range"
        " (high - low) has been compressed over a recent period."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily range
        history = (
            history.with_columns(
                (pl.col("high") - pl.col("low")).alias("range")
            )
            .sort("session_date", descending=False)
            .with_column(pl.col("range").rank(method="dense", descending=True).alias("rank"))
        )

        # Filter symbols with high range compression
        compressed_symbols = (
            history.group_by("symbol")
            .agg(
                (pl.col("range").sum()).alias("total_range"),
                (pl.col("rank").min()).alias("min_rank"),
            )
            .filter((pl.col("min_rank") <= 1) & (pl.col("total_range") > 0))
        )

        if compressed_symbols.height < 5:
            return Signal(information_available_at=stamp, weights={})

        # Select top symbols based on range compression
        picks = [row["symbol"] for row in compressed_symbols.sort(
            "min_rank", descending=True).head(5)]
        weight = 1.0 / len(picks)
        return Signal(information_available_at=stamp, weights={s: weight for s in picks})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
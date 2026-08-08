from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "During periods of range compression, the volatility between high and low prices "
        "within a window tends to be lower than average. This suggests that the market is "
        "consolidating before potentially breaking out in one direction. Entering positions "
        "during such times can capture gains when breakout happens."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_groups = (
            history
            .group_by("symbol")
            .agg(
                pl.col("high").max().alias("max_high"),
                pl.col("low").min().alias("min_low"),
                (pl.col("close") - pl.col("adj_close").shift(1)).abs().sum().alias("total_range"),
            )
        )

        range_ratio = symbol_groups.select(
            ((pl.col("max_high") - pl.col("min_low")) / pl.col("total_range")).alias("range_ratio")
        )

        top_symbols = range_ratio.sort("range_ratio", descending=True).select(pl.col("symbol").head(5)).to_list()[0]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
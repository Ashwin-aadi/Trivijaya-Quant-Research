from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that the market is consolidating, and may soon break out. "
        "By identifying symbols with reduced price volatility, we can potentially find opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily range for each symbol
        ranges = (
            history.select(
                pl.col("symbol"),
                (pl.col("high") - pl.col("low")).alias("range")
            )
            .group_by("symbol")
            .agg(pl.col("range").sum().alias("total_range"))
        )

        # Normalize range by the window
        ranges = (
            ranges.with_columns(
                (pl.col("total_range") / self._window).alias("avg_daily_range")
            )
        )

        # Find symbols with minimized range
        min_avg_ranges = ranges.sort("avg_daily_range").select(pl.col("symbol"))
        if min_avg_ranges.height == 0:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = [row[0] for row in min_avg_ranges.to_dict(as_series=False).values()]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression5d(Strategy):
    rationale = (
        "Dispersion in daily range (high - low) over a 5-day lookback period can indicate "
        "potential market instability or consolidation. A significant decrease in the range "
        "can suggest increased stability and potential for trend continuation."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        daily_ranges = (
            history.sort("session_date")
            .group_by("symbol")
            .agg(
                (pl.col("high") - pl.col("low")).alias("range"),
                pl.count().alias("count"),
            )
        )

        filtered_daily_ranges = daily_ranges.filter(pl.col("count") == self._window)

        if filtered_daily_ranges.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_range = (
            filtered_daily_ranges.select(
                (pl.col("range").mean().alias("mean_range"))
            )
            .collect()
            .row(0)[0]
        )

        if not mean_range:
            return Signal(information_available_at=stamp, weights={})

        compressed_symbols = (
            daily_ranges.filter(pl.col("range") < 1.5 * mean_range)
            .select(["symbol"])
            .to_pandas()["symbol"]
            .tolist()
        )

        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in compressed_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
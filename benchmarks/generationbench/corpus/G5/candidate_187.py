from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that the market is consolidating before a potential breakout. "
        "High dispersion in daily price ranges can indicate accumulation or distribution activity, "
        "potentially leading to a significant move."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily range
        high_low_diff = (history["high"] - history["low"]).alias("range")
        history = history.with_column(high_low_diff)

        # Compute mean and standard deviation of the range over the window
        mean_range = history.select(pl.col("range").mean().alias("mean_range"))[0]["mean_range"]
        std_dev_range = history.select(pl.col("range").std().alias("std_range"))[0]["std_range"]

        # Identify symbols with a range compression and potential breakout conditions
        compressed_symbols = (
            history.with_columns(
                (pl.col("range") - mean_range).abs() / std_dev_range > self._threshold,
                (pl.col("close").shift(-1) - pl.col("close")).alias("close_change"),
            )
            .group_by("symbol")
            .agg(
                pl.count().alias("count"),
                (
                    pl.col("close_change") >= 0.5 * std_dev_range
                ).sum()
                .alias("potential_breakout_count"),
            )
            .filter((pl.col("count") > 0) & (pl.col("potential_breakout_count") > 0))
            .select(["symbol"])
        )["symbol"].to_list()

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
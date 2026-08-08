from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when price volatility decreases, suggesting a potential "
        "uptrend. A strategy based on this phenomenon could exploit the reversion to higher "
        "volatility and trend strength after such periods."
    )

    def __init__(self, window: int = 20, threshold: float = 0.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily range for each symbol
        ranges = (
            history.with_columns(
                (pl.col("high") - pl.col("low")).alias("range")
            )
            .group_by("symbol")
            .agg(pl.col("range").mean().alias("avg_range"))
        )

        # Compute the ratio of current daily range to average range
        signals = (
            history.join(ranges, on="symbol", how="left")
            .with_columns(
                (pl.col("high") - pl.col("low")) / pl.col("avg_range").alias("range_ratio")
            )
            .select(["session_date", "symbol", "range_ratio"])
        )

        # Filter for symbols where the range ratio is below a threshold
        compressed = signals.filter(pl.col("range_ratio") < self._threshold)

        if compressed.height == 0:
            return Signal(information_available_at=stamp, weights={})

        # Pick top N symbols with highest compression
        top_symbols = compressed.sort("range_ratio", descending=True).head(5)
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols["symbol"].to_list()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression20d(Strategy):
    rationale = (
        "Range compression indicates that prices are moving within a narrower band than usual, "
        "suggesting increased volatility and potentially higher returns. By identifying symbols with the most significant range compression, we can capture these opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily range
        high_low_diff = (
            history.select(
                pl.col("symbol"),
                (pl.col("high") - pl.col("low")).alias("range")
            ).group_by("symbol").agg(pl.col("range").max().alias("max_range"))
        )
        recent_highs = view.closes(lookback=self._window).select(
            pl.col("session_date"), pl.col(view.symbols[0]).alias("recent_high")
        )

        # Join to get the most recent high for each symbol
        recent_highs = (
            history.join(recent_highs, on="symbol", how="inner")
            .select(pl.col("symbol"), "range", "recent_high")
        )
        
        # Calculate range compression as a percentage of historical max range
        compressed_range = (recent_highs.with_columns(
            ((pl.col("recent_high") - pl.col("high")) / pl.col("max_range")).alias("compression_ratio")
        ).sort("compression_ratio").select(pl.col("symbol"), "compression_ratio"))

        # Filter and select top N symbols with the highest range compression
        compressed_range = (
            compressed_range.head(self._window)
            .with_column(
                (pl.col("compression_ratio") / self._window).alias("weight")
            )
        )

        if compressed_range.height == 0:
            return Signal(information_available_at=stamp, weights={})

        # Convert to dictionary of symbol: weight
        top_symbols = compressed_range.select(pl.col("symbol"), "weight").to_dict(as_series=False)

        return Signal(
            information_available_at=stamp,
            weights=top_symbols
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
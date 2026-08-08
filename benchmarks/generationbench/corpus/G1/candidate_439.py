from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "During periods of range compression, volatility is low and there may be a "
        "tendency for prices to breakout in either direction. Identifying such periods can "
        "help capture significant moves."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily range for each symbol
        ranges = (
            history.select(
                pl.col("symbol"), (pl.col("high") - pl.col("low")).alias("range")
            )
            .group_by("symbol")
            .agg(pl.col("range").mean().alias("avg_range"))
        )

        # Calculate the compression factor for each symbol
        compressed = (
            history.join(ranges, on="symbol", how="left")
            .with_columns(
                (pl.col("high") - pl.col("low")) / pl.col("avg_range").fill_null(1.0).alias("compression_factor")
            )
            .select(["symbol", "session_date", "compression_factor"])
        )

        # Identify symbols with high compression
        compressed = (
            compressed.sort("compression_factor", descending=True)
            .filter(pl.col("compression_factor") > self._threshold)
            .head(self._window)
        )

        if compressed.height < 1:
            return Signal(information_available_at=stamp, weights={})

        # Assign equal weight to each selected symbol
        symbols = [row["symbol"] for row in compressed.to_dict(as_pandas=False)["symbol"]]
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
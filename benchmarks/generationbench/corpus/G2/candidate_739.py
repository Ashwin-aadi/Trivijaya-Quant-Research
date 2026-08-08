from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that prices are trading within a narrower range than "
        "historical norms. This suggests market sentiment is neutral or weak, and could signal "
        "a potential breakout in either direction. By identifying stocks with compressed ranges, "
        "we can position ourselves to benefit from the subsequent price movement."
    )

    def __init__(self, window: int = 60, compression_threshold: float = 0.1) -> None:
        self._window = window
        self._compression_threshold = compression_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the daily range for each symbol
        ranges = (
            history.select([
                pl.col("symbol"),
                (pl.col("high") - pl.col("low")).alias("range")
            ])
            .group_by("symbol")
            .agg(pl.col("range").sum().alias("total_range"))
            .sort("total_range", descending=False)
        )

        # Calculate the mean range over the window
        mean_range = ranges.select(
            (pl.col("total_range") / self._window).alias("mean_range")
        ).item()

        # Identify symbols with range compression
        compressed_symbols = [
            row["symbol"] for i, row in ranges.iter_rows()
            if row["range"] < mean_range * (1 - self._compression_threshold)
        ]

        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in compressed_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
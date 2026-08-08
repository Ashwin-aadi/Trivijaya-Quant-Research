from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression often indicates a period of consolidation where prices are moving "
        "within a narrow range. This can be followed by breakout or reversal. By identifying "
        "symbols with high range compression, we may find opportunities to enter positions."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        range_compression = (
            history.group_by("symbol")
            .agg(
                pl.col("high").max() - pl.col("low").min().alias("range"),
                (pl.col("high").max() / pl.col("low").min()).alias("ratio"),
            )
            .sort("range", descending=True)
            .with_columns(
                ((pl.col("close") - pl.col("open")) / pl.col("adj_close").shift(1)).alias(
                    "daily_change"
                ),
                (pl.col("high") == history["close"].max()).alias("top_of_range"),
                (pl.col("low") == history["close"].min()).alias("bottom_of_range"),
            )
        )

        range_compression = (
            range_compression.filter(
                (pl.col("range").is_not_null())
                & (pl.col("ratio") > 1.0)
                & ((pl.col("top_of_range")) | (pl.col("bottom_of_range")))
            )
            .sort("range", descending=True)
            .head(5)
        )

        if range_compression.is_empty():
            return Signal(information_available_at=stamp, weights={})

        top_5_symbols = [row["symbol"] for row in range_compression.to_dicts()]
        weight_per_symbol = 1.0 / len(top_5_symbols)

        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight_per_symbol
                for symbol in top_5_symbols
                if symbol in view.symbols
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
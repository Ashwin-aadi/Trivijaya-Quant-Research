from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the price is consolidating, which can precede "
        "a breakout or a reversal. By identifying symbols with reduced volatility, we aim to "
        "capture potential breakout opportunities."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate range compression
        range_compression = (
            history.group_by("symbol")
            .agg(
                (pl.col("high").max() - pl.col("low").min()).alias("range"),
                (pl.col("close") / pl.col("open").shift(1) - 1.0).alias("return"),
            )
            .with_columns(
                ((pl.col("range") / pl.col("return")) * 100.0).alias("compression_ratio")
            )
        )

        # Filter out symbols with no non-zero returns or ranges
        range_compression = (
            range_compression.filter(
                (pl.col("range") > 0) & (pl.col("close") / pl.col("open").shift(1) - 1.0).is_not_null()
            )
        )

        if range_compression.height == 0:
            return Signal(information_available_at=stamp, weights={})

        # Sort and select top N symbols based on compression ratio
        top_n_symbols = [
            symbol for _, row in range_compression.sort("compression_ratio", descending=True).iter_rows()
            if float(row["compression_ratio"]) > 0.1 * max(range_compression["compression_ratio"].to_list())
        ]

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
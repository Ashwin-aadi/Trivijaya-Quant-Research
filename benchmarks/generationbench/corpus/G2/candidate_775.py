from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression signals high volatility and potential for mean reversion. "
        "Stocks that have recently experienced less price movement compared to their recent range are "
        "likely to experience a sudden increase in price activity."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily range for each symbol
        ranges = (
            history.select(
                pl.col("session_date"), 
                (pl.col("high") - pl.col("low")).alias("range")
            )
            .sort("session_date", descending=False)
            .group_by("symbol")
            .agg(pl.col("range").mean().alias("avg_range"))
        )

        # Calculate the average range for each symbol over the window period
        avg_ranges = (
            ranges.select(
                pl.col("symbol"), 
                (pl.col("avg_range") / pl.col("avg_range").shift(1) - 1.0).alias("range_ratio")
            )
            .sort("range_ratio", descending=True)
            .head(self._window * len(view.symbols))
        )

        # Select symbols with the highest range compression
        compressed_symbols = [symbol for symbol in avg_ranges["symbol"].to_list() if abs(avg_ranges.filter(pl.col("symbol") == symbol)["range_ratio"].item()) > 0.1]

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
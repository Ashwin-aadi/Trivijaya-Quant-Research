from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates a market is consolidating and may soon breakout. "
        "By identifying symbols with reduced price volatility over the past period, we can "
        "find potentially high-momentum candidates for entry."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily range for each symbol
        history = (
            history
            .with_columns(
                (pl.col("high") - pl.col("low")).alias("daily_range")
            )
            .group_by("symbol")
            .agg(
                pl.col("daily_range").mean().alias("avg_daily_range"),
                (pl.col("daily_range").rank(method="ordinal", descending=True)).alias("range_rank")
            )
        )

        # Filter symbols with the lowest average daily range
        filtered_symbols = history.filter(
            (pl.col("range_rank") <= 5)
        ).select(["symbol"])

        if filtered_symbols.is_empty():
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_symbols["symbol"].to_list())
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol in filtered_symbols["symbol"].to_list()
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
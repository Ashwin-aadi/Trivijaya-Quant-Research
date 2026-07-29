from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates a period of reduced volatility. This suggests that the "
        "market is consolidating and could be setting up for a breakout or continuation in the "
        "existing trend. By identifying symbols with high range compression, we can capture "
        "potential opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate the range for each symbol
        history = (
            history.with_columns(
                (pl.col("high") - pl.col("low")).alias("range")
            )
            .group_by("symbol")
            .agg(pl.col("range").mean().alias("avg_range"))
            .sort("avg_range", descending=False)
            .filter(pl.col("avg_range") > 0)  # Ensure non-zero range
        )

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in history.to_dicts()[-self._window:]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
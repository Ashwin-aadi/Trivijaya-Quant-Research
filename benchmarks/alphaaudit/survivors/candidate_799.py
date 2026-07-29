from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates a period of reduced volatility and can precede "
        "a breakout or trend reversal. By identifying symbols with the highest range "
        "compression over a given window, we can capture potential breakout opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate the high-low range for each symbol
        history = (
            history.with_columns(
                (pl.col("high") - pl.col("low")).alias("range")
            )
            .group_by("symbol")
            .agg(pl.col("range").max().alias("max_range"))
            .sort("max_range", descending=True)
        )

        # Select top N symbols with the highest range compression
        picks: list[str] = history["symbol"].to_list()[:5]
        
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
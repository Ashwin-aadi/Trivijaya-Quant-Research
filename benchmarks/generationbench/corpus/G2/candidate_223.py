from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "During periods of low volatility or range compression, asset prices may move more "
        "dramatically once volatility increases. Identifying such periods can provide a "
        "window for profitable trades."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the daily range for each symbol
        ranges = (
            history.select(
                [
                    pl.col("symbol"),
                    (pl.col("high") - pl.col("low")).alias("range"),
                    pl.col("session_date")
                ]
            )
            .group_by("symbol", "session_date")
            .agg(pl.col("range").mean().alias("avg_range"))
        )

        # Calculate the average range for each symbol over the window
        avg_ranges = (
            ranges.groupby("symbol")
            .agg((pl.col("avg_range") / pl.col("avg_range").shift(1) - 1.0).alias("ratio"))
            .sort("symbol", descending=True)
        )

        # Identify symbols with significantly reduced range compression
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in avg_ranges.columns or pl.col("symbol").is_null().any():
                continue
            ratio = float(avg_ranges[avg_ranges["symbol"] == symbol]["ratio"].to_list()[0])
            if ratio < 0.5:  # A simple threshold for reduced range compression
                picks.append(symbol)

        pick_count = len(picks)
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / pick_count
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
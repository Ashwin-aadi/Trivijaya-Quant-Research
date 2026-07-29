from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates reduced volatility and increased price concentration. "
        "During such periods, stocks may exhibit less extreme daily movements but maintain "
        "the potential for significant trends. Identifying these periods can provide opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily range for each symbol
        ranges = (
            history.select(
                pl.col("symbol"),
                (pl.col("high") - pl.col("low")).alias("range")
            )
            .group_by("symbol")
            .agg(pl.col("range").mean().alias("avg_range"))
            .sort("avg_range", descending=False)
        )

        # Filter out symbols with very low average range
        filtered_symbols = ranges.filter(
            (pl.col("avg_range") > 0.5) & (pl.col("avg_range") < 2.0)
        ).select("symbol").to_series().to_list()

        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in filtered_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
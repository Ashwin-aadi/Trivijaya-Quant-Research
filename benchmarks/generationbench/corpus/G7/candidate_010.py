from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalVolumeBreakout(Strategy):
    rationale = (
        "Seasonal effects can significantly influence trading volumes in short-term "
        "markets. By analyzing the volume data over a 3-month lookback period, we can "
        "identify periods of high activity and potentially capitalize on them."
    )

    def __init__(self, window: int = 90, max_positions: int = 20) -> None:
        self._window = window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_changes = (
            history.select(
                pl.col("symbol"),
                (pl.col("volume") / pl.col("volume").shift(self._window) - 1.0).alias("volume_change")
            )
            .group_by("symbol")
            .agg(pl.col("volume_change").mean().alias("avg_volume_change"))
        )

        if volume_changes.height < self._max_positions:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = (
            volume_changes.sort("avg_volume_change", descending=True)
            .select(pl.col("symbol"))
            .to_series()
            .to_list()[:self._max_positions]
        )

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression30d(Strategy):
    rationale = (
        "High dispersion in daily price ranges indicates periods of increased volatility. "
        "Entering positions when such periods are observed aims to capture gains during "
        "market expansions while exiting based on trailing stop-loss helps manage risk."
    )

    def __init__(self, window: int = 30, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        high_low_range = (
            history.select(
                pl.col("symbol"),
                (pl.col("high") - pl.col("low")).alias("range")
            )
            .group_by("symbol")
            .agg(pl.col("range").mean().alias("avg_range"))
            .sort("avg_range", descending=True)
            .head(self._top_n)
        )

        picks: list[str] = [row["symbol"] for row in high_low_range.rows()]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
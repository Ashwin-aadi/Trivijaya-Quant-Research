from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the market is consolidating, and may be setting up "
        "for a breakout. By identifying symbols with reduced price volatility over a period, "
        "we can identify potential candidates for future strong moves."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        min_max_spread = (
            history.group_by("symbol")
            .agg(
                (pl.col("high").min() - pl.col("low").max()).alias("spread"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
            )
            .sort("spread", descending=False)
        )

        if min_max_spread.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        compressed_symbols = [
            symbol for _, row in min_max_spread.iter_rows() if abs(row["spread"]) <= 0.1
        ][:20]

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
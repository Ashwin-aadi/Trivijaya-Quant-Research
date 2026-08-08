from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression20d(Strategy):
    rationale = (
        "Range compression measures the volatility of daily prices. High dispersion in "
        "trading ranges suggests that the market is consolidating or preparing for a breakout, "
        "providing opportunities for traders to enter positions."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        range_compression = (
            history.select([
                pl.col("symbol"),
                (pl.col("high") - pl.col("low")).alias("range")
            ])
            .group_by("symbol")
            .agg(pl.col("range").mean().alias("avg_range"))
            .sort("avg_range", descending=True)
            .select(["symbol", "avg_range"])
        )

        if range_compression.height < 1:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in range_compression.to_dicts()[:5]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
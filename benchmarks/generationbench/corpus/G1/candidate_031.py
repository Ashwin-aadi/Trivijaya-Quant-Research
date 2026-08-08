from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the price action is becoming more volatile and "
        "that a breakout or reversal may be imminent. By identifying symbols with high "
        "range compression, we can opportunistically enter positions before any potential "
        "trend change."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        range_compression = (
            history.select(
                pl.col("symbol"),
                (pl.col("high") - pl.col("low")).alias("range"),
                (pl.col("adj_close").shift(-1) - pl.col("adj_close")).abs().alias("close_change"),
            )
            .with_columns(
                ((pl.col("range") / pl.col("close_change")) * 100).alias("ratio")
            )
            .sort("symbol", descending=True)
            .group_by("symbol")
            .agg(pl.col("ratio").mean().alias("avg_ratio"))
        )

        if range_compression.height < view.symbols.size():
            return Signal(information_available_at=stamp, weights={})

        top_symbols = range_compression.sort("avg_ratio", descending=True)["symbol"].to_list()[:5]
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
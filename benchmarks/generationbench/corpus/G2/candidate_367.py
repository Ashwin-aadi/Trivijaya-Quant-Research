from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that when volatility in a stock decreases, the price "
        "may start to move more dramatically. By identifying stocks where range has been "
        "compressed recently, we can anticipate potential breakout opportunities."
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
                (pl.col("high") - pl.col("low")).alias("range_width")
            )
            .sort("session_date", descending=True)
            .group_by("symbol")
            .agg(pl.col("range_width").mean().alias("avg_range"))
        )

        latest_closes = view.closes(lookback=self._window)
        latest_highs = view.history(lookback=self._window).select(
            pl.col("symbol"), "high"
        )
        latest_lows = view.history(lookback=self._window).select(
            pl.col("symbol"), "low"
        )

        range_compression = range_compression.join(latest_closes, on="symbol", how="inner")
        range_compression = range_compression.join(latest_highs, on="symbol", how="inner")
        range_compression = range_compression.join(latest_lows, on="symbol", how="inner")

        range_compression = (
            range_compression.with_columns(
                ((pl.col("high") - pl.col("low")) / pl.col("avg_range")).alias("ratio"),
                (1.0 - (pl.col("range_width") / pl.col("avg_range"))).alias("compression")
            )
            .sort("compression", descending=True)
            .select(["symbol", "ratio"])
        )

        top_n = min(range_compression.height, 5)
        selected_symbols = range_compression["symbol"].to_list()[:top_n]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
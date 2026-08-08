from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates reduced volatility and potentially increased predictability "
        "in the market. By identifying stocks with higher range compression, we can exploit this "
        "period of market consolidation."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        range_compression = (
            history.group_by("symbol")
            .agg(
                (pl.col("high") - pl.col("low")).alias("range"),
                ((pl.col("close").shift(-1) - pl.col("open")) / 2).alias("mid_close_open"),
                (pl.col("adj_close") - pl.col("adj_close").shift(1)).abs().mean().alias("avg_price_change")
            )
            .with_columns(
                (((pl.col("range") + pl.col("mid_close_open")).is_null()).cast(pl.int8).sum() == 0)
                & (pl.col("avg_price_change") < pl.col("range").shift(1) / 2).alias("valid_range_compression")
            )
            .filter((pl.col("close") > pl.col("open")) & (pl.col("valid_range_compression")))
            .select(pl.col("symbol"))
        )

        if range_compression.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks = [str(symbol) for symbol in range_compression["symbol"].to_list()[:5]]
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
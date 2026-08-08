from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "During periods of range compression, volatility is low and there may be more "
        "opportunities for profitable trades. We aim to identify such periods and allocate "
        "capital accordingly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_high = history.groupby("symbol").agg(pl.col("high").mean().alias("mean_high"))
        mean_low = history.groupby("symbol").agg(pl.col("low").mean().alias("mean_low"))

        range_width = (mean_high - mean_low).with_columns(
            ((pl.col("mean_high") / pl.col("mean_low")) - 1.0).alias("range_ratio")
        )

        if range_width.is_empty():
            return Signal(information_available_at=stamp, weights={})

        compressed_symbols = (
            range_width.sort("range_ratio", descending=False)
            .select(["symbol"])
            .head(5)["symbol"]
            .to_list()
        )
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
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompressionStrategy(Strategy):
    rationale = (
        "Range compression indicates that the market is consolidating and preparing for a "
        "potential breakout. This strategy aims to identify stocks with high range compression "
        "indicating strong buying or selling pressure."
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
                pl.col("high").max() - pl.col("low").min().alias("range")
            )
            .with_columns((pl.col("range") / pl.col("adj_close").shift(1) * 100).alias("compression"))
            .sort("session_date", descending=False)
        )

        if range_compression.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_n_symbols = range_compression["symbol"].to_list()[-self._window :]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight
                for symbol in set(top_n_symbols) & view.symbols
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
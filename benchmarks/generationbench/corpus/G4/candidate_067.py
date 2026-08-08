from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompressionDispersion(Strategy):
    rationale = (
        "This strategy identifies sectors experiencing range compression or dispersion by "
        "analyzing the volatility of stock prices. Long positions are taken in stocks showing "
        "range compression, anticipating a breakout, while short positions are initiated in "
        "stocks with high dispersion to profit from potential narrowing of price ranges."
    )

    def __init__(self, lookback: int = 60, long_top_n: int = 10, short_top_n: int = 10) -> None:
        self._lookback = lookback
        self._long_top_n = long_top_n
        self._short_top_n = short_top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate High-Low range
        ranges = (
            history.lazy()
            .group_by("symbol")
            .agg(
                pl.col("high").max() - pl.col("low").min().alias("range"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
            )
            .collect()
        )

        # Filter out symbols not in the current universe
        ranges = (
            ranges.filter(pl.col("symbol").is_in(view.symbols))
            .select(["symbol", "range", "return"])
            .with_columns((pl.col("range") / pl.sum("range").over()).alias("rank_range"))
            .sort("rank_range")
        )

        # Identify long and short candidates
        long_symbols = ranges.head(self._long_top_n)["symbol"].to_list()
        short_symbols = ranges.tail(self._short_top_n)["symbol"].to_list()

        weights: dict[str, float] = {}
        if long_symbols:
            weight = 0.5 / len(long_symbols)
            for symbol in long_symbols:
                weights[symbol] = weight

        if short_symbols:
            weight = -0.5 / len(short_symbols)
            for symbol in short_symbols:
                weights[symbol] = weight

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "This strategy aims to exploit mean-reverting behavior in stock prices relative "
        "to a trailing reference level. It seeks to capitalize on price deviations from "
        "historical averages by taking long or short positions based on the z-score of the "
        "current price relative to its 20-day moving average."
    )

    def __init__(self, window: int = 20, top_n: int = 50, position_size: float = 0.02) -> None:
        self._window = window
        self._top_n = top_n
        self._position_size = position_size

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate 20-day moving average
        ma_20 = (
            closes.select(
                pl.col("adj_close").rolling_mean(self._window).alias(f"ma_{self._window}")
            )
            .group_by("symbol")
            .agg(pl.all().last())
        )

        # Merge with original close data to get relative deviation
        merged = closes.join(ma_20, on="symbol", how="inner")
        z_scores = (merged["adj_close"] - merged[f"ma_{self._window}"]) / (
            pl.col("adj_close").std()
        )
        merged = merged.with_column(z_scores.alias("z_score"))

        # Filter and sort by absolute z-score
        filtered = merged.filter(pl.col("z_score").is_not_null())
        ranked = (
            filtered.sort("z_score", descending=False)
            .with_columns(
                (pl.arange(1, pl.count() + 1) / pl.count()).alias("rank")
            )
            .select(["symbol", "z_score", "rank"])
        )

        # Select top N stocks based on z-score
        top_stocks = ranked.head(self._top_n).to_dict(as_series=False)

        if not top_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight_per_stock = self._position_size / len(top_stocks)
        weights = {symbol: weight_per_stock for symbol in top_stocks["symbol"]}

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
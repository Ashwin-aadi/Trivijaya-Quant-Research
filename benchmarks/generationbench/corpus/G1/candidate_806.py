from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends that have historically shown higher volatility. "
        "The idea is that such trends are more likely to continue than those with lower volatility."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        df = history.select(
            pl.col("symbol"),
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
            (pl.col("adj_close") - pl.col("adj_close").shift(1)).abs().alias("abs_return"),
        )

        volatility = df.groupby("symbol").agg(
            (pl.col("abs_return").mean()).alias("volatility")
        )
        trend_strength = df.groupby("symbol").agg(
            (
                (pl.col("return") * pl.col("return")).sum()
                / self._window
                .cast(pl.Float64)
            ).alias("trend_strength")
        )

        combined = volatility.join(trend_strength, on="symbol")
        ranked = combined.sort("volatility", descending=True).select(
            "symbol", "trend_strength"
        )
        top_symbols = [row[0] for row in ranked.head(self._window).rows()]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

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
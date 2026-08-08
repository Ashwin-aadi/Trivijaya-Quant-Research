from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for market confidence and tradability. By equal-weighting "
        "high-liquidity stocks, we aim to capture the benefits of liquid markets without "
        "over-concentrating in any single stock."
    )

    def __init__(self, liquidity_threshold: float = 1_000_000, top_n: int = 20) -> None:
        self._threshold = liquidity_threshold
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.filter(pl.col("volume") > self._threshold)
            .group_by("symbol")
            .agg(
                pl.count().alias("count"),
                pl.col("adj_close").mean().alias("avg_close"),
            )
            .sort(by="count", descending=True)
            .head(self._top_n)
        )

        symbols = liquidity_screened["symbol"].to_list()
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
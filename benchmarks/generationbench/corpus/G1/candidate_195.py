from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screening ensures that only stocks with sufficient trading volume are "
        "considered for the portfolio. Equal weighting across selected stocks simplifies risk "
        "management and can provide a balanced exposure to the market."
    )

    def __init__(self, liquidity_threshold: int = 1_000_000) -> None:
        self._threshold = liquidity_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        filtered_history = (
            history.filter(
                (pl.col("symbol").is_in(view.symbols))
                & (pl.col("volume") > self._threshold)
            )
            .group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("avg_close"))
            .sort("avg_close", descending=True)
            .head(10)
        )

        if filtered_history.height < 1:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / filtered_history.height
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol in filtered_history.select("symbol").to_list()[0]
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
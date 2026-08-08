from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is often a proxy for market efficiency and quality of information. "
        "Highly liquid stocks are more likely to be correctly priced and less prone to sudden price swings. "
        "By focusing on the most liquid stocks, this strategy aims to capture the benefits of market efficiency."
    )

    def __init__(self, liquidity_window: int = 30) -> None:
        self._window = liquidity_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.select(pl.col("symbol"), pl.col("volume").sum().alias("total_volume"))
            .group_by("symbol")
            .agg(pl.col("total_volume").mean().alias("avg_volume"))
            .sort("avg_volume", descending=True)
            .head(self._window)["symbol"]
            .to_list()
        )

        if not liquidity_scores:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(liquidity_scores)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in liquidity_scores},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
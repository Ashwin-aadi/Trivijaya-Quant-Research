from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity is a key driver of returns. By equal-weighting stocks based on their liquidity, "
        "we aim to capture the benefits of higher trading volumes and lower transaction costs."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.select(pl.col("symbol"))
            .group_by("symbol")
            .agg(
                (pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity_score")
            )
            .sort("liquidity_score", descending=True)
            .head(self._window)["symbol"]
        )

        liquidity_scores = [str(symbol) for symbol in liquidity_scores]
        weights = {s: 1.0 / len(liquidity_scores) for s in liquidity_scores}
        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
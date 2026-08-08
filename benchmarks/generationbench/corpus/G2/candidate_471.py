from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Higher liquidity stocks are typically more efficient in price discovery and trading, "
        "potentially leading to reduced market impact costs and higher liquidity. By equally "
        "weighting the most liquid stocks, we aim to benefit from these characteristics."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.groupby("symbol")
                   .agg((pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity_score"))
                   .sort("liquidity_score", descending=True)
                   .head(self._window)["symbol"]
        )

        if not liquidity_scores.height:
            return Signal(information_available_at=stamp, weights={})

        n = len(liquidity_scores)
        weight = 1.0 / n
        weights = {symbol: weight for symbol in liquidity_scores.to_list()}
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
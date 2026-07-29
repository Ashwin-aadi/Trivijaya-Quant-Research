from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Equally weighting the most liquid stocks can capture market-wide movements while "
        "leveraging the higher trading volumes to potentially reduce transaction costs and "
        "market impact."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = history.select(
            pl.col("symbol"),
            (pl.col("volume").sum() / 10_000).alias("liquidity_score"),
        )
        top_symbols = (
            liquidity_scores.sort("liquidity_score", descending=True)
            .head(self._window)["symbol"]
            .to_list()
        )

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_per_symbol = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_symbol for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
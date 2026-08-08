from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity-weighted equal weighting involves allocating capital based on the liquidity "
        "of stocks. More liquid assets are given more weight to ensure that trades do not significantly impact their price."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_list = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbol_list:
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity_score"),
            )
            .sort(by="liquidity_score", descending=True)
            .select(["symbol", "liquidity_score"])
            .to_dicts()
        )

        if not liquidity_scores:
            return Signal(information_available_at=stamp, weights={})

        total_liquidity = sum(score["liquidity_score"] for score in liquidity_scores)
        weight_per_symbol = {
            score["symbol"]: score["liquidity_score"] / total_liquidity
            for score in liquidity_scores[:self._window]
        }

        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_symbol[s] if s in weight_per_symbol else 0.0 for s in view.symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
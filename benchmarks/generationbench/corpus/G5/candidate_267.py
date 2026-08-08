from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity-screened equal weighting involves selecting the most liquid stocks "
        "and assigning them equal weights. This strategy aims to reduce transaction costs and "
        "improve trade execution while maintaining a diversified portfolio."
    )

    def __init__(self, liquidity_window: int = 20) -> None:
        self._liquidity_window = liquidity_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._liquidity_window)
        if history.is_empty() or history.height < self._liquidity_window * 2:
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.groupby("symbol")
            .agg(
                pl.col("volume").sum().alias("total_volume"),
            )
            .sort("total_volume", descending=True)
            .head(10)["symbol"]
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
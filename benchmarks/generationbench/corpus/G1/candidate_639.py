from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Screening for stocks with high liquidity ensures we are focusing on assets "
        "with sufficient trading volume to execute large orders without significant impact. "
        "An equal-weight portfolio across the most liquid stocks provides a balanced approach."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.select(["symbol", "volume"])
            .group_by("symbol")
            .agg(pl.col("volume").sum().alias("total_volume"))
            .sort("total_volume", descending=True)
            .head(self._window)["symbol"]
            .to_list()
        )

        if not liquidity_screened:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(liquidity_screened)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in liquidity_screened},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Higher liquidity stocks tend to have lower bid-ask spreads and are less prone to "
        "abnormal price movements. By equal-weighting the most liquid stocks, we aim to "
        "benefit from reduced transaction costs and smoother price action."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.groupby("symbol")
                   .agg(pl.col("volume").sum().alias("total_volume"))
                   .sort("total_volume", descending=True)
                   .head(self._window)["symbol"]
                   .to_list()
        )

        if not liquidity_scores:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(liquidity_scores)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in liquidity_scores}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
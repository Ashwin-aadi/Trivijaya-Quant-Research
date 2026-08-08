from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Equally weighting stocks based on their liquidity can capture the idea that more "
        "liquid assets are less likely to be mispriced and may exhibit higher returns over "
        "the long run. This strategy aims to allocate capital in a way that gives preference "
        "to stocks with higher trading volumes."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.select(
                pl.col("symbol"), (pl.col("volume").sum() / self._window).alias("avg_vol")
            )
            .sort("avg_vol", descending=True)
            .head(20)["symbol"]
        )

        if not liquidity_screened.to_list():
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(liquidity_screened.to_list())
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
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are more likely to have efficient pricing and lower trading "
        "costs. By equal-weighting high-liquidity stocks, we can exploit the positive correlation "
        "between liquidity and returns."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.group_by("symbol")
            .agg(
                pl.col("volume").mean().alias("avg_volume"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
            )
            .with_columns((pl.col("avg_volume") * pl.col("return")).alias("weighted_return"))
        )

        if liquidity_scores.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = (
            liquidity_scores.sort("weighted_return", descending=True)
            .select(["symbol"])
            .head(10)["symbol"]
            .to_list()
        )
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
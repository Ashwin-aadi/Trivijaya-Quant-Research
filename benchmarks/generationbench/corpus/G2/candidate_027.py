from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a key driver of market efficiency. Highly liquid stocks are more likely "
        "to have prices that reflect their true intrinsic value and less susceptible to price "
        "distortions due to small trades. By equal-weighting the most liquid stocks, we can "
        "potentially benefit from this liquidity premium."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.select(
                pl.col("symbol"),
                (pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity_score")
            )
            .group_by("symbol")
            .agg(pl.col("liquidity_score").mean().alias("avg_liquidity_score"))
            .sort("avg_liquidity_score", descending=True)
        )

        if liquidity_scores.height < 5:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row[0] for row in liquidity_scores.to_pandas().head(5).values]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity is a key driver of stock performance. By equal-weighting stocks based on "
        "their liquidity, we aim to capture the benefits of higher trading volume and more "
        "efficient pricing."
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
                pl.col("symbol"), (pl.col("volume").sum() / 20).alias("liquidity_score")
            )
            .group_by("symbol")
            .agg(pl.col("liquidity_score").mean().alias("mean_volume"))
            .sort("mean_volume", descending=True)
        )

        if liquidity_scores.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        symbols = [row["symbol"] for row in liquidity_scores.to_dicts()]
        weight = 1.0 / min(len(symbols), 5)  # Limit to top 5 symbols
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols[:5]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
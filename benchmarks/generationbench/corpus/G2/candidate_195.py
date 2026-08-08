from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are often more efficient in price discovery and trading. "
        "By equal-weighting the most liquid names, we can capitalize on this efficiency to "
        "potentially achieve higher returns with lower transaction costs."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.group_by("symbol")
                   .agg((pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity_score"))
                   .sort("liquidity_score", descending=True)
                   .select(["symbol", "liquidity_score"])
        )

        top_symbols = liquidity_scores.to_pandas().head(10)["symbol"].tolist()

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        equal_weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: equal_weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are more likely to have efficient pricing and less susceptible "
        "to market anomalies. By equally weighting high-liquidity stocks, the strategy aims to "
        "capitalize on the stability and predictability of these securities."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.groupby("symbol")
                   .agg((pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity_score"))
                   .sort("liquidity_score", descending=True)
        )

        if liquidity_scores.height < 5:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row[0] for row in liquidity_scores.head(5).iter_rows()]
        weight = 1.0 / len(top_symbols)

        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
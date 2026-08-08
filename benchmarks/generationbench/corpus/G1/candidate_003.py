from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "We screen for high liquidity stocks by selecting the top 30 most traded symbols. "
        "Then we equally weight these selected stocks across the portfolio to capture"
        "their trading volume without overconcentration risk."
    )

    def __init__(self, lookback: int = 20) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.group_by("symbol")
            .agg(pl.col("volume").mean().alias("avg_volume"))
            .sort("avg_volume", descending=True)
            .head(30)
        )

        symbols = [str(row["symbol"]) for row in liquidity_screened.to_dicts()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_per_symbol = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_symbol for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
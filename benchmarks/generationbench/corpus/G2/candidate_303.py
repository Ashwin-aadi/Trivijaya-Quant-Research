from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Higher liquidity stocks are generally more efficient and less prone to market anomalies. "
        "By equally weighting these stocks, the strategy can benefit from reduced idiosyncratic risk."
    )

    def __init__(self, lookback: int = 30) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.select(["symbol", "volume"])
            .group_by("symbol")
            .agg(pl.col("volume").sum().alias("total_volume"))
            .sort("total_volume", descending=True)
            .limit(10)  # Select top 10 most liquid stocks
        )

        if liquidity_screened.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [row["symbol"] for row in liquidity_screened.iter_rows()]
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
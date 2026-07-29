from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for market efficiency and confidence. By selecting the "
        "most liquid assets, we aim to reduce trading costs and improve execution quality."
    )

    def __init__(self, window: int = 20) -> None:
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
            .head(self._window)
        )

        if liquidity_screened.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [row["symbol"] for row in liquidity_screened.to_dicts()]
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
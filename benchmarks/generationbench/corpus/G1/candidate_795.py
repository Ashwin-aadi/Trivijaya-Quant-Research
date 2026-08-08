from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy aims to equally weight a subset of highly liquid stocks. "
        "High liquidity indicates lower transaction costs and more reliable price discovery."
    )

    def __init__(self, window: int = 20, liquidity_threshold: float = 1_000_000) -> None:
        self._window = window
        self._liquidity_threshold = liquidity_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Filter by liquidity
        high_liquidity_symbols = (
            history.lazy()
            .group_by("symbol")
            .agg(pl.col("volume").sum().alias("total_volume"))
            .filter(pl.col("total_volume") > self._liquidity_threshold)
            .select(["symbol"])
            .collect()["symbol"]
            .to_list()
        )

        if not high_liquidity_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Equal weighting
        weight = 1.0 / len(high_liquidity_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in high_liquidity_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screening ensures that only stocks with high trading volume are "
        "considered for the portfolio. Equal weighting across these high-liquidity stocks "
        "helps in reducing concentration risk and improving diversification."
    )

    def __init__(self, min_volume: float = 10_000_000) -> None:
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter for high liquidity
        high_liquidity_symbols = (
            history.lazy()
            .group_by("symbol")
            .agg(pl.col("volume").sum())
            .filter(pl.col("volume_sum") > self._min_volume)
            .select("symbol")
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
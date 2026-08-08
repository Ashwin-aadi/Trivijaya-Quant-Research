from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for marketability and thus tradability. "
        "High-liquidity stocks can be entered or exited without significantly impacting their price. "
        "An equal-weighted portfolio of highly liquid stocks is likely to have lower transaction costs."
    )

    def __init__(self, liquidity_threshold: float = 10_000_000, lookback_days: int = 365) -> None:
        self._threshold = liquidity_threshold
        self._lookback = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_liquidity_symbols = (
            history.groupby("symbol")
            .agg(pl.col("volume").sum().alias("total_volume"))
            .filter(pl.col("total_volume") > self._threshold)
            .select("symbol")
            .to_dict(True)["symbol"]
        )

        if not high_liquidity_symbols:
            return Signal(information_available_at=stamp, weights={})

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
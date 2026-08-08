from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy screens for highly liquid stocks and equal weights them to "
        "create a diversified portfolio. Highly liquid assets are less prone to price "
        "distortions due to trading."
    )

    def __init__(self, liquidity_threshold: float = 1_000_000) -> None:
        self._liquidity_threshold = liquidity_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=60)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquid_symbols = (
            history.groupby("symbol")
                   .agg(pl.col("volume").sum().alias("total_volume"))
                   .filter(pl.col("total_volume") > self._liquidity_threshold)
                   .select("symbol")
                   .to_dict()[0]
        )

        if not liquid_symbols:
            return Signal(information_available_at=stamp, weights={})

        n = len(liquid_symbols)
        weight = 1.0 / n
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in liquid_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
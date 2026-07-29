from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for market efficiency and trading volume. "
        "High-liquidity stocks are more likely to be fair-priced and less prone to significant price swings. "
        "This strategy aims to screen for high liquidity before applying equal weighting among the selected assets."
    )

    def __init__(self, window: int = 20, min_volume_threshold: float = 1_000_000) -> None:
        self._window = window
        self._min_volume_threshold = min_volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_liquidity_symbols = (
            history.select(["symbol", "volume"])
                   .filter(pl.col("volume").gt(self._min_volume_threshold))
                   .group_by("symbol")
                   .agg(pl.col("volume").mean().alias("avg_volume"))
                   .sort("avg_volume", descending=True)
                   .select("symbol")
                   .to_series()
                   .to_list()[:20]
        )

        if not high_liquidity_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(high_liquidity_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in high_liquidity_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
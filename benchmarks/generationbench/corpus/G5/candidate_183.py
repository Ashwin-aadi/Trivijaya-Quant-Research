from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity screening ensures that only stocks with sufficient trading volume are "
        "considered. Equal weighting among these stocks promotes diversification and can "
        "reduce the impact of individual stock-specific risks."
    )

    def __init__(self, window: int = 20, liquidity_threshold: float = 10_000_000) -> None:
        self._window = window
        self._liquidity_threshold = liquidity_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Filter symbols by liquidity
        liquidity_screened = (
            history
            .group_by("symbol")
            .agg(pl.col("volume").mean().alias("avg_volume"))
            .filter(pl.col("avg_volume") > self._liquidity_threshold)
        )

        symbols = [str(row["symbol"]) for row in liquidity_screened.select("symbol").rows()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        # Equal weighting among filtered symbols
        weights = {s: 1.0 / len(symbols) for s in symbols}
        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
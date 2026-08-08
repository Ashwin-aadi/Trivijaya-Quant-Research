from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy aims to equal-weight a subset of the market with high liquidity, "
        "reducing transaction costs and ensuring that no single stock dominates the portfolio."
    )

    def __init__(self, min_trading_volume: float = 100_000) -> None:
        self._min_trading_volume = min_trading_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter by minimum trading volume
        high_liquidity_symbols = (
            history.select(["symbol", "volume"])
            .group_by("symbol")
            .agg(pl.col("volume").mean().alias("avg_volume"))
            .filter(pl.col("avg_volume") > self._min_trading_volume)
            .select("symbol")
            .to_series()
        )

        if high_liquidity_symbols.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Equal-weight the selected symbols
        num_symbols = len(high_liquidity_symbols)
        weight = 1.0 / num_symbols
        weights = {symbol: weight for symbol in high_liquidity_symbols.to_list()}

        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
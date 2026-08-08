from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is often a proxy for market efficiency and can indicate that stocks are "
        "traded more frequently, potentially leading to better execution and lower costs. "
        "A strategy that screens for high liquidity before applying equal weighting may "
        "benefit from reduced transaction costs and improved price discoverability."
    )

    def __init__(self, min_volume: int = 10_000) -> None:
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=252)  # Using 252 trading days for robustness
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_volume_symbols = (
            history.select(["symbol", "session_date", "volume"])
            .filter(pl.col("volume") > self._min_volume)
            .group_by("symbol")
            .agg(pl.count().alias("trading_days"))
            .sort("trading_days", descending=True)
            .head(10)  # Select top 10 symbols
            .select(["symbol"])
        )
        
        if high_volume_symbols.is_empty():
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(high_volume_symbols["symbol"].to_list())
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight
                for symbol in high_volume_symbols["symbol"].to_list()
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
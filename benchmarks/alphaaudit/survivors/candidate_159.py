from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for marketability and can indicate the ease of trading shares. "
        "By equal-weighting stocks with sufficient liquidity, we aim to mitigate the risk of "
        "illiquid assets that could significantly impact our portfolio's performance."
    )

    def __init__(self, min_trading_volume: float = 100_000) -> None:
        self._min_trading_volume = min_trading_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=None)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        trading_volumes = (
            history.select(pl.col("symbol"), pl.col("volume"))
            .group_by("symbol")
            .agg(pl.col("volume").sum().alias("total_volume"))
            .sort("total_volume", descending=True)
        )

        symbols_with_sufficient_liquidity = (
            trading_volumes.filter(pl.col("total_volume") > self._min_trading_volume)
            .select(["symbol"])
            .to_series()
            .to_list()
        )

        if not symbols_with_sufficient_liquidity:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_with_sufficient_liquidity)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol in symbols_with_sufficient_liquidity
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
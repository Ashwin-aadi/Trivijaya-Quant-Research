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

        high_liquidity_symbols = (
            history.select(["symbol", "volume"])
            .filter(pl.col("volume") > self._min_trading_volume)
            .select("symbol")
            .distinct()
            .to_dict(as_series=False)["symbol"]
        )

        if not high_liquidity_symbols:
            print(f"No symbols meet the minimum trading volume of {self._min_trading_volume}.")
            return Signal(information_available_at=stamp, weights={})

        equal_weight = 1.0 / len(high_liquidity_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: equal_weight for symbol in high_liquidity_symbols
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
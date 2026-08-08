from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Highly liquid stocks tend to have higher trading volume and lower bid-ask spreads, "
        "potentially leading to better execution of trades. By equal-weighting the most liquid "
        "stocks in the market, we can capture the benefits of liquidity without overconcentrating "
        "in any single stock."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity = (
            history.filter(pl.col("symbol").is_in(view.symbols))
            .group_by("symbol")
            .agg(
                pl.sum("volume").alias("total_volume"),
                (pl.col("adj_close") - pl.col("open")).abs().sum().alias("price_volatility"),
            )
        )

        # Equal-weight the top n most liquid symbols, where liquidity is defined as total volume
        sorted_symbols = (
            liquidity.sort("total_volume", descending=True)
            .select(["symbol", "total_volume"])
            .to_series()
        ).to_list()[: self._window]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights=dict(zip(sorted_symbols, [weight] * len(sorted_symbols))),
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
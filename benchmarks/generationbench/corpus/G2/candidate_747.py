from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Higher liquidity stocks are generally more efficient in their pricing and less prone to "
        "price anomalies. By focusing on highly liquid stocks, we can potentially achieve more "
        "stable returns with less risk."
    )

    def __init__(self, lookback: int = 20) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.group_by("symbol")
            .agg(
                (pl.col("volume").sum()).alias("total_volume"),
                (pl.col("adj_close") / pl.col("open")).mean().alias("price_change_ratio"),
            )
            .sort(
                "total_volume", descending=True
            )  # Higher volume indicates higher liquidity
            .head(10)  # Top 10 most liquid stocks
        )

        if liquidity_screened.height < 10:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / 10
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol in liquidity_screened["symbol"].to_list()
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
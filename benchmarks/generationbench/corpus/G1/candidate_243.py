from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a key indicator of market interest and can help identify stocks that are "
        "actively traded. By equally weighting these liquid stocks, the strategy aims to capture "
        "the collective performance of highly traded equities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.filter(pl.col("volume").shift(-1).is_not_null())
            .group_by("symbol")
            .agg(
                pl.col("adj_close").mean().alias("avg_price"),
                pl.col("volume").sum().alias("total_volume"),
            )
            .sort("total_volume", descending=True)
            .head(self._window)
        )

        if liquidity_screened.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(liquidity_screened)
        weights = {
            symbol: weight for symbol in liquidity_screened["symbol"].to_list()
        }
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
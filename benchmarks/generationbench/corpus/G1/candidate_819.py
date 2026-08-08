from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity is a critical factor in trading. High liquidity stocks can be traded without "
        "significantly moving the price, making them attractive for portfolio construction."
    )

    def __init__(self, lookback_window: int = 20) -> None:
        self._lookback_window = lookback_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.filter(pl.col("symbol").is_in(view.symbols))
            .group_by("symbol")
            .agg(
                pl.col("volume").sum().alias("total_volume"),
            )
            .sort("total_volume", descending=True)
            .head(self._lookback_window)["symbol"]
        )

        if liquidity_scores.height < self._lookback_window:
            return Signal(information_available_at=stamp, weights={})

        equal_weight = 1.0 / self._lookback_window
        weight_dict = {symbol: equal_weight for symbol in liquidity_scores}
        return Signal(
            information_available_at=stamp,
            weights=weight_dict,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
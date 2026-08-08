from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedStrategy(Strategy):
    rationale = (
        "Liquidity is a proxy for market efficiency and demand. High liquidity suggests "
        "greater willingness to trade and potentially higher returns. By equal-weighting "
        "high-liquidity stocks, we aim to capture the benefits of these more actively traded "
        "stocks."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened_history = (
            history.group_by("symbol")
                   .agg((pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity_score"))
                   .sort("liquidity_score", descending=True)
                   .head(self._window + 1)  # Include the symbol itself
        )

        symbols = liquidity_screened_history["symbol"].to_list()
        weights = {s: 1.0 / len(symbols) for s in symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
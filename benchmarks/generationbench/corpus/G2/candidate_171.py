from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for market efficiency and accessibility. Higher liquidity "
        "typically means better price discovery and reduced slippage costs for large trades. "
        "This strategy aims to identify the most liquid stocks, which can be expected to have "
        "more stable prices and potentially lower trading costs."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history
            .group_by("symbol")
            .agg((pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity_score"))
        )
        sorted_symbols = liquidity_scores.sort("liquidity_score", descending=True)["symbol"].to_list()
        
        if len(sorted_symbols) < 5:
            return Signal(information_available_at=stamp, weights={})

        top_liquid_symbols = sorted_symbols[:5]
        weight = 1.0 / len(top_liquid_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_liquid_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
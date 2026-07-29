from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy selects the most liquid stocks based on volume and then "
        "equal weights them in the portfolio. Liquid assets are less prone to price manipulation "
        "and can be traded more freely without significant impact on their prices."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened_symbols = (
            history.select(["symbol", "volume"])
            .group_by("symbol")
            .agg(pl.col("volume").mean().alias("avg_volume"))
            .sort("avg_volume", descending=True)
            .head(10)  # Select top 10 most liquid symbols
            .get_column("symbol")
            .to_list()
        )

        if not liquidity_screened_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(liquidity_screened_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol in liquidity_screened_symbols
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
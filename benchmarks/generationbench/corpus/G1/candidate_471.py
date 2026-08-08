from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy selects the most liquid stocks from the NIFTY 100 and applies equal "
        "weighting to them. Higher liquidity is often associated with better marketability, "
        "potentially leading to more stable prices and reduced transaction costs."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.groupby("symbol")
                   .agg((pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity_score"))
                   .sort("liquidity_score", descending=True)
                   .head(10)  # Select top 10 most liquid stocks
        )

        symbols = liquidity_scores.select(pl.col("symbol")).to_list()[0]
        weight_per_symbol = 1.0 / len(symbols)

        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_symbol for s in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Highly liquid stocks are more likely to be priced efficiently and thus provide "
        "better risk-adjusted returns. This strategy selects the most liquid stocks based on "
        "average daily volume over a specified window."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.group_by("symbol")
            .agg((pl.col("volume").mean().alias("avg_volume")))
            .sort("avg_volume", descending=True)
            .head(self._window)
        )

        if liquidity_screened.is_empty():
            return Signal(information_available_at=stamp, weights={})

        weights = (
            liquidity_screened.with_columns(pl.lit(1.0 / self._window).alias("weight"))
            .select(["symbol", "weight"])
            .to_dict(False)
        )
        
        return Signal(information_available_at=stamp, weights={s: w for s, w in weights})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
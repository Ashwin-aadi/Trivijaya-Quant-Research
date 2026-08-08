from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are more likely to have their prices reflect fundamental "
        "information quickly and accurately. By equally weighting high liquidity stocks, we can "
        "potentially benefit from faster price movements in response to market events."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity = _calculate_liquidity(history)
        symbols = [s for s in view.symbols if s in liquidity.columns and not liquidity[s].is_null().any()]
        
        if len(symbols) < 5:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_liquidity(history: pl.DataFrame) -> pl.DataFrame:
    volume = history.select(pl.col("volume").sum().alias("total_volume"))
    average_daily_volume = (history.group_by("symbol")
                            .agg((pl.col("volume") / pl.col("session_date").count()).alias("avg_daily_vol")))
    
    return volume.join(average_daily_volume, on="symbol", how="inner")
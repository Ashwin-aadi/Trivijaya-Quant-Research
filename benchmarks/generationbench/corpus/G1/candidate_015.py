from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for market confidence and tradability. By focusing on highly "
        "liquid stocks, we aim to minimize the impact of our trades on the market price and "
        "maximize overall portfolio efficiency."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        liquidity = history.select(
            pl.col("symbol").alias("SYMBOL"),
            (pl.col("volume") / 240.0).alias("avg_volume")
        ).group_by("SYMBOL").agg(pl.col("avg_volume").mean().alias("avg_volume")).sort("avg_volume", descending=True)

        if liquidity.height < len(symbols):
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [s for _, s in liquidity.iter_rows(nrows=len(symbols))]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
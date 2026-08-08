from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Assets that outperform their peers over a defined period are likely to continue "
        "outperforming due to superior fundamentals or market sentiment. This strategy "
        "identifies such assets based on relative strength."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .group_by("symbol")
            .agg(pl.sum("r").alias("total_return"))
            .sort("total_return", descending=True)
            .select(["symbol", "total_return"])
        )

        # Filter out symbols with insufficient data
        history = history.filter((pl.col("total_return") >= self._window) & (pl.col("total_return") < float("inf")))

        if history.height < 5:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in history.head(5)]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
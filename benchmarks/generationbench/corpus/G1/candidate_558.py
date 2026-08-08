from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Identifying stocks with strong relative performance can help in capturing "
        "momentum and outperformance. This strategy selects the top-performing stocks "
        "relative to the market index over a lookback period."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(sym) for sym in view.symbols]
        avg_market_close = (
            history.groupby("session_date")
                   .agg(pl.col("adj_close").mean().alias("avg_adj_close"))
                   .select(["session_date", "avg_adj_close"])
                   .with_columns(
                       (pl.col("adj_close") / pl.col("avg_adj_close") - 1.0).alias("strength")
                   )
        )

        strong_stocks = (
            history
                   .group_by("symbol")
                   .agg(pl.col("strength").max().alias("max_strength"))
                   .select(["symbol", "max_strength"])
                   .sort("max_strength", descending=True)
                   .head(5)
        )

        if strong_stocks.height < 5:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(strong_stocks)
        strong_symbols = [row["symbol"] for row in strong_stocks.to_dicts()]
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in strong_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
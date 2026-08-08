from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityBasedStrategy(Strategy):
    rationale = (
        "This strategy exploits seasonal patterns in specific sectors such as tourism and agriculture. "
        "It identifies favorable periods to enter the market and exits based on the waning of these effects or general market conditions."
    )

    def __init__(self, window: int = 365, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        recent_history = history.filter(pl.col("session_date") >= date(2020, 1, 1))
        tourism_weights = recent_history.with_columns(
            pl.when(
                (pl.col("symbol").is_in(["TOURISM-SECTOR-TICKERS"])) & (
                    pl.col("adj_close").rolling_max(window=365) == pl.col("close")
                )
            ).then(1.0)
        ).group_by("symbol").agg(pl.count()).sort("count", descending=True).to_dict(
            "vertical"
        )["count"]

        agriculture_weights = recent_history.with_columns(
            pl.when(
                (pl.col("symbol").is_in(["AGRICULTURE-SECTOR-TICKERS"])) & (
                    pl.col("adj_close").rolling_max(window=365) == pl.col("close")
                )
            ).then(1.0)
        ).group_by("symbol").agg(pl.count()).sort("count", descending=True).to_dict(
            "vertical"
        )["count"]

        weights = {**tourism_weights, **agriculture_weights}
        sorted_weights = {k: v for k, v in sorted(weights.items(), key=lambda item: -item[1])}

        picks = list(sorted_weights.keys())[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
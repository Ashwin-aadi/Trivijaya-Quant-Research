from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends based on volatility. It identifies symbols with high "
        "volatility over a short term and buys them, expecting continuation of the trend."
    )

    def __init__(self, window: int = 10, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or len(history["session_date"].unique()) < self._window:
            return Signal(information_available_at=stamp, weights={})

        recent_closes = history.select(
            pl.col("symbol"),
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
        )

        symbol_volatility = (
            recent_closes.groupby("symbol")
                        .agg(pl.col("return").std().alias("volatility"))
                        .sort(by="volatility", descending=True)
                        .select("symbol")
                        .to_series()
                        .to_list()[: self._top_n]
        )

        if not symbol_volatility:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbol_volatility)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbol_volatility},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest
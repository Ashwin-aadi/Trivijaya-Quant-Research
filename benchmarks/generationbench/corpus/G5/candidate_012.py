from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "recently to continue outperforming. This strategy buys top performers and sells bottom "
        "performers over a short-term horizon."
    )

    def __init__(self, window: int = 20, top_n: int = 5, bottom_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n
        self._bottom_n = bottom_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.latest_close()
        symbols = set(latest_closes.keys()).intersection(set(history["symbol"].to_list()))

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        recent_returns = (
            history.filter(pl.col("session_date") >= (view.as_of - pl.duration(days=self._window)))
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
        )

        if recent_returns.is_empty():
            return Signal(information_available_at=stamp, weights={})

        top_symbols = (
            recent_returns.sort("avg_return", descending=True)
            .select(["symbol"])
            .head(self._top_n)
            .column("symbol")
            .to_list()
        )
        bottom_symbols = (
            recent_returns.sort("avg_return")
            .select(["symbol"])
            .head(self._bottom_n)
            .column("symbol")
            .to_list()
        )

        weights = {s: 0.2 for s in top_symbols}
        if self._top_n + self._bottom_n < len(top_symbols) + len(bottom_symbols):
            return Signal(information_available_at=stamp, weights={})

        for s in bottom_symbols:
            if len(weights) < self._top_n + self._bottom_n:
                weights[s] = -0.1
            else:
                break

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
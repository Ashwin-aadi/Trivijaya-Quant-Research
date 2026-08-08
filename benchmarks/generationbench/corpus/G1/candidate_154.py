from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Certain stocks in the Indian market exhibit strong seasonal trends. By identifying "
        "these patterns, we can predict buying opportunities during favorable seasons."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter for NIFTY 100 symbols
        nifty_symbols = tuple(s for s in view.symbols if s.startswith("NIFTY"))
        filtered_history = history.select(
            pl.col("symbol").filter(pl.col("symbol").is_in(nifty_symbols))
        )

        # Calculate the average return over the window period
        avg_returns = (
            filtered_history.group_by("symbol")
            .agg((pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("avg_return"))
            .sort("avg_return", descending=True)
            .head(5)["symbol"]
            .to_list()
        )

        if not avg_returns:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(avg_returns)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in avg_returns},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
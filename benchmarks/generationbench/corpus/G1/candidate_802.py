from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Historical data often show that stock performance can be influenced by seasonal patterns. "
        "This strategy exploits such patterns to identify potential trading opportunities."
    )

    def __init__(self, window: int = 30, seasonality_window: int = 60) -> None:
        self._window = window
        self._seasonality_window = seasonality_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._seasonality_window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the average daily returns
        daily_returns = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .filter(pl.col("session_date") < stamp)
            .sort("session_date", descending=False)
            .select(["symbol", "session_date", "r"])
        )

        # Group by symbol and calculate the average return for each
        avg_returns = daily_returns.groupby("symbol").agg(
            pl.col("r").mean().alias("avg_return")
        )
        seasonally_high_symbols = (
            avg_returns.sort(pl.col("avg_return"), descending=True)
            .head(self._window)
            .select(["symbol"])
            .to_dict(as_series=False)["symbol"]
        )

        if not seasonally_high_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(seasonally_high_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in seasonally_high_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
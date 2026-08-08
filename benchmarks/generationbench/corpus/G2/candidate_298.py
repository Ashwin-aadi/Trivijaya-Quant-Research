from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "Certain sectors or industries may exhibit predictable patterns in their performance "
        "based on calendar effects. For instance, tourism-related stocks might show higher returns "
        "during the festive season. By identifying these seasonal trends, we can capitalize on "
        "predictable market movements."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Extract the month from the session date for seasonal analysis
        history = (
            history.with_columns(
                pl.col("session_date").dt.month().alias("month")
            )
            .group_by("symbol", "month")
            .agg(pl.col("adj_close").mean().alias("avg_monthly_return"))
            .sort("month")
        )

        # Calculate the average return for each month over the entire period
        avg_returns = history.group_by("month").agg(
            pl.col("avg_monthly_return").mean().alias("avg_monthly_return")
        ).collect()

        # Identify months with above-average returns
        recent_avg_return = (
            view.history(lookback=self._window).select(pl.col("session_date").dt.month()).to_series()
        )
        recent_returns = history.filter(
            pl.col("month").is_in(recent_avg_return.to_list())
        ).select("avg_monthly_return")

        above_average_months = recent_returns.height > 0

        if not above_average_months:
            return Signal(information_available_at=stamp, weights={})

        # Select symbols with returns in the identified months
        symbols_with_above_avg_returns = (
            history.filter(pl.col("month").is_in(recent_avg_return.to_list()))
            .select("symbol")
            .to_series()
            .unique()
            .to_list()
        )

        weight = 1.0 / len(symbols_with_above_avg_returns)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols_with_above_avg_returns},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
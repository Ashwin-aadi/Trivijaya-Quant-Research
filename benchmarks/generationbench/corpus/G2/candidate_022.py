from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Certain times of the year may historically exhibit higher returns due to seasonal "
        "effects. For instance, in India, some sectors might perform better during monsoon or "
        "post-monsoon periods. This strategy aims to capture such seasonality by identifying "
        "periods with historical outperformance and allocating capital accordingly."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter historical data to get the latest 30 days
        history = history.sort("session_date").tail(self._window)

        # Group by month and calculate mean returns for each month
        grouped = (
            history.group_by("session_date.dt.month()")
            .agg(
                (pl.col("close") / pl.col("open").shift(1) - 1.0).alias("return")
            )
            .with_columns((pl.col("return").mean().alias("mean_return")))
        )

        # Identify the month with the highest mean return
        top_month = grouped.sort("mean_return", descending=True)["session_date.dt.month()"].first()

        # Get symbols that are active in the identified month
        symbols_in_top_month = history.select(pl.col("symbol")).filter(
            pl.col("session_date").dt.month() == top_month
        ).unique().to_list()[0]

        if not symbols_in_top_month:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_in_top_month)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in symbols_in_top_month},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
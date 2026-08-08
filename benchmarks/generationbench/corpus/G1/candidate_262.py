from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality in the Indian market can be driven by various factors such as religious "
        "festivals, government policies, and macroeconomic events. This strategy aims to capture "
        "the momentum during specific times of the year that historically have shown positive returns."
    )

    def __init__(self, window: int = 20, seasonality_window: int = 365) -> None:
        self._window = window
        self._seasonality_window = seasonality_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._seasonality_window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Identify the current season
        today = date.fromordinal(stamp.toordinal())
        year_start = today.replace(month=1, day=1)
        seasonal_period = (today - year_start).days // 30 + 1

        # Extract relevant historical data for the same period in previous years
        filtered_history = (
            history.with_columns(
                pl.col("session_date").dt.year().alias("year"),
                pl.col("session_date").dt.weekday().alias("weekday"),
            )
            .filter((pl.col("session_date") >= year_start)
                    & (pl.col("session_date") < today))
            .group_by(["symbol", "weekday"])
            .agg(
                (
                    (pl.col("adj_close").mean()).alias("mean_price"),
                    ((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).sum())
                    .alias("log_return_sum")
                )
            )
            .sort(["symbol", "weekday"])
        )

        # Calculate the average log returns for each symbol in the current season
        seasonal_trends = (
            filtered_history
            .group_by("symbol")
            .agg(
                pl.col("log_return_sum").mean().alias("seasonal_log_return_mean"),
                pl.col("mean_price").max().alias("max_price")
            )
            .sort("seasonal_log_return_mean", descending=True)
        )

        top_symbols = seasonal_trends["symbol"].head(self._window).to_list()

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Allocate equally across the top symbols
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol in top_symbols
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
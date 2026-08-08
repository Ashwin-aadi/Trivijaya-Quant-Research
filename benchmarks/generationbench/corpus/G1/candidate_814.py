from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Equities often exhibit seasonal patterns in their returns. By identifying and "
        "capitalizing on these patterns, we can potentially achieve higher returns during "
        "favorable seasons."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter out symbols that do not have enough data
        symbols_with_data = [symbol for symbol in view.symbols if symbol in history.columns]
        if len(symbols_with_data) < 5:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the average close price for each symbol over the lookback period
        closes = history.select(
            pl.col("session_date"), *[pl.col(symbol).alias(f"{symbol}_close") for symbol in symbols_with_data]
        ).group_by("session_date").agg(
            [pl.mean(pl.col(f"{symbol}_close")).alias(f"avg_{symbol}_close") for symbol in symbols_with_data]
        )

        # Identify the season of each date
        def get_season(date_str: str) -> int:
            month = int(date_str.split("-")[1])
            return (month - 3) // 3 + 1

        closes = closes.with_columns(
            pl.col("session_date").str.strptime(pl.Date).apply(get_season).alias("season")
        )

        # Group by season and calculate the average close price
        seasonal_data = (
            closes.group_by("season")
            .agg([pl.mean(f"avg_{symbol}_close") for symbol in symbols_with_data])
            .collect()
        )
        
        # Find the highest average closing price across all seasons
        max_avg_price = seasonal_data.height > 0
        if not max_avg_price:
            return Signal(information_available_at=stamp, weights={})

        season_of_max_avg = int(seasonal_data.sort("avg_close", descending=True).select(["season"]).to_list()[0][0])
        symbols_in_max_season = [symbol for symbol in symbols_with_data if get_season(history[history["session_date"] == stamp].column("session_date").to_list()[0]) == season_of_max_avg]

        # Allocate weights to the symbols that are currently in the max season
        weight_per_symbol = 1.0 / len(symbols_in_max_season)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight_per_symbol for s in symbols_in_max_season}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
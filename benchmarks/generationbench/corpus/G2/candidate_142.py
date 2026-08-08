from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Certain months of the year may historically have seen stronger performance in the "
        "Indian equity market due to seasonal effects such as harvest cycles, government policies, "
        "or holidays that affect consumer behavior and corporate earnings."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Filter the symbols to only those that have enough data
        valid_symbols = [symbol for symbol in view.symbols if symbol in closes.columns]
        if not valid_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the mean close price for each month across all symbols
        monthly_closes = (
            closes[valid_symbols].select("session_date", *valid_symbols)
            .with_columns(pl.col(valid_symbols).mean().suffix("_mean"))
            .rename({"_mean": "monthly_mean"})
            .group_by(pl.col("session_date").dt.month())
            .agg([pl.col("monthly_mean").alias(f"mean_{month}") for month in range(1, 13)])
        )

        # Find the months with highest and lowest mean close prices
        monthly_means = (
            monthly_closes.select(*[f"mean_{month}" for month in range(1, 13)])
            .select(pl.all().max(), pl.all().min())
        )
        highest_mean = float(monthly_means["mean_1"].max())
        lowest_mean = float(monthly_means["mean_12"].min())

        # Identify symbols that have closed above the mean in their strongest month
        strong_symbols: list[str] = []
        for symbol in valid_symbols:
            monthly_close = closes.select(pl.col(symbol).mean_by("session_date").map_dict(lambda x: x[0]).alias("monthly_close"))
            if float(monthly_close[1]) > highest_mean or float(monthly_close[-1]) < lowest_mean:
                strong_symbols.append(symbol)

        # If no symbols are identified, return an empty Signal
        if not strong_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(strong_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in strong_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
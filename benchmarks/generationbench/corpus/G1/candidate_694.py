from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Certain stocks in the Indian market exhibit stronger performance during specific "
        "seasons. By identifying these seasonal patterns, we can allocate capital to those "
        "stocks at the start of their peak periods."
    )

    def __init__(self, window: int = 30, lookback: int = 12) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback).sort("session_date")
        if history.height < self._lookback + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width != len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Identify the season for the current date
        month = stamp.month

        # Calculate seasonal performance by year
        avg_returns_by_month = (
            history.group_by("session_date").agg(
                (pl.col("close") / pl.col("adj_close").shift(self._window) - 1).alias("return")
            )
            .with_columns((pl.col("return").mean().over(pl.col("session_date").dt.month()).alias("avg_monthly_return")))
            .sort("session_date", descending=False)
            .group_by("month_of_year").agg(
                pl.col("avg_monthly_return").mean().alias("avg_return")
            )
        )

        # Find the top performing month
        top_month = avg_returns_by_month.order_by(pl.col("avg_return"), descending=True).select("month_of_year")[0]
        symbol_weights: dict[str, float] = {}

        # Allocate capital to stocks in the identified top-performing month
        for symbol in view.symbols:
            if str(month) == history.select("session_date").dt.month().shift(-self._window).filter(pl.col("session_date") < stamp).tail(1)[0]:
                symbol_weights[symbol] = 1.0 / len(view.symbols)

        return Signal(information_available_at=stamp, weights=symbol_weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
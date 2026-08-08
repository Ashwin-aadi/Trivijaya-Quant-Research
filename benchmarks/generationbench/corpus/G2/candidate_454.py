from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality in equity markets can arise from a variety of factors, including "
        "government policies, weather patterns, and cultural events. By identifying stocks that "
        "tend to perform better during specific times of the year, we can capture these seasonal "
        "patterns for potential profit."
    )

    def __init__(self, window: int = 365) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_signals = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            history = view.history().filter(pl.col("session_date") <= stamp).select(
                "symbol", "session_date", pl.col("adj_close")
            ).collect()
            if history.is_empty():
                continue

            monthly_closes = history.groupby(pl.col("session_date").dt.month()).agg(
                pl.col("adj_close").mean().alias("avg_monthly_close")
            )
            current_month_avg = (
                monthly_closes.filter(pl.col("session_date") == stamp.month)
                .select("avg_monthly_close")
                .to_list()[0][0]
            )

            if not isinstance(current_month_avg, float):
                continue

            rank = closes.filter(
                pl.col(symbol).is_not_null()
            ).sort(
                (pl.col(symbol) / pl.col(symbol).shift(1)).alias("return"),
                descending=True
            ).with_column((pl.col(symbol) / pl.col(symbol).shift(1) - 1.0).alias("return")).select(
                pl.col("session_date").dt.month().alias("month"), "return"
            ).groupby(pl.col("month")).agg(
                pl.col("return").mean().alias("monthly_return")
            ).collect()

            monthly_ranks = rank.sort("monthly_return", descending=True)
            seasonal_signals[symbol] = 1.0 / (len(rank) + 1 - monthly_ranks.filter(
                pl.col("session_date").dt.month() == stamp.month
            ).select(pl.col("month")).to_list()[0][0])

        if not seasonal_signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(seasonal_signals.values())
        adjusted_weights = {s: w / total_weight for s, w in seasonal_signals.items()}
        return Signal(
            information_available_at=stamp, weights=adjusted_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
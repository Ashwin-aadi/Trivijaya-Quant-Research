from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality can provide predictive power in equity markets due to recurring "
        "patterns related to specific times of the year. We exploit these patterns by "
        "identifying symbols that historically perform well during certain months."
    )

    def __init__(self, lookback_years: int = 2) -> None:
        self._lookback_years = lookback_years

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_years * 365)

        if history.height < (self._lookback_years + 1) * 365:
            return Signal(information_available_at=stamp, weights={})

        # Group by month and symbol to calculate mean close for each
        grouped = (
            history.select(
                pl.col("symbol"),
                pl.col("session_date").dt.month().alias("month"),
                pl.col("adj_close").alias("close"),
            )
            .group_by(["symbol", "month"])
            .agg(pl.col("close").mean())
            .sort("symbol")
        )

        # Find symbols that have a high close in the last month of the lookback period
        symbol_month_pairs = (
            grouped.filter(pl.col("month") == history["session_date"].dt.month().max())
            .select(["symbol"])
            .to_series()
            .to_list()
        )

        if not symbol_month_pairs:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbol_month_pairs)
        return Signal(
            information_available_at=stamp,
            weights={
                s: weight
                for s in symbol_month_month_pairs
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
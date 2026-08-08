from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityStrategy(Strategy):
    rationale = (
        "This strategy exploits historical seasonality effects in the Indian equity market by "
        "buying stocks that have historically outperformed during specific months and selling them "
        "after they have potentially reached their peak. The primary focus is on January, which often"
        " exhibits strong returns due to the 'January effect'."
    )

    def __init__(self, lookback_years: int = 5, top_n: int = 20) -> None:
        self._lookback_years = lookback_years
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_years * 365)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter data to focus on the last year for current context
        start_date = stamp - pl.duration.Years(self._lookback_years)
        recent_history = history.filter((pl.col("session_date") >= start_date) & (pl.col("session_date") < stamp))

        symbol_counts = recent_history.select(
            [pl.col("symbol"), pl.col("session_date").dt.month_name().alias("month")]
        ).group_by("symbol", "month").count()

        # Focus on January
        january_data = symbol_counts.filter(pl.col("month") == "January")
        if january_data.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Compute mean returns for each stock in December of the previous year
        december_returns = recent_history.filter(
            (pl.col("session_date").dt.month_name() == "December") & (pl.col("session_date") < stamp - pl.duration.Months(1))
        ).group_by("symbol").agg(pl.col("adj_close").mean().alias("december_mean"))

        # Join to get December means for stocks that had strong January returns
        ranked_stocks = january_data.join(december_returns, on="symbol", how="inner").sort(
            pl.col("count"), descending=True
        ).head(self._top_n)

        weights = {row["symbol"]: 1.0 / len(ranked_stocks) for row in ranked_stocks.iter_rows()}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
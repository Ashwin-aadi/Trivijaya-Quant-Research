from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality effects suggest that certain times of the year have historically provided "
        "better returns than others due to predictable changes in economic conditions, such as "
        "holiday spending or company-specific events. By identifying and exploiting these trends, "
        "we can potentially generate alpha."
    )

    def __init__(self, window: int = 30, seasonality_window: int = 90) -> None:
        self._window = window
        self._seasonality_window = seasonality_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._seasonality_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Compute the seasonal trends
        closes = history.select(
            pl.col("session_date").dt.month().alias("month"),
            "adj_close"
        ).pivot(index="month", columns="symbol", values="adj_close")

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue

            monthly_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(monthly_closes) < self._seasonality_window // 12:
                continue

            # Calculate the average return for each month over the seasonality window
            avg_returns: list[float] = []
            for i in range(1, 13):
                monthly_data = [monthly_closes[(j - 1) * 12 + (i - 1)] for j in range(1, self._seasonality_window // 12 + 1)]
                if len(monthly_data) >= self._window:
                    avg_return = sum([(value / monthly_data[j-1] - 1.0) for j, value in enumerate(monthly_data)]) / len(monthly_data)
                    avg_returns.append(avg_return)

            # Identify the month with the highest average return
            best_month_idx = max(range(len(avg_returns)), key=avg_returns.__getitem__)
            if avg_returns[best_month_idx] > 0.0:
                return Signal(
                    information_available_at=stamp,
                    weights={symbol: 1.0 for symbol in view.symbols}
                )

        return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
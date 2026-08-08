from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality in stock markets refers to consistent price movements that occur at specific times of the year. "
        "For instance, certain sectors may perform well during the holiday season due to increased consumer spending. "
        "By identifying these trends, we can predict and exploit them for trading opportunities."
    )

    def __init__(self, window: int = 365, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width == 0:
            return Signal(information_available_at=stamp, weights={})

        seasonal_trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate the mean and standard deviation of daily returns over the window period
            daily_returns = [(values[i] - values[i-1]) / values[i-1] if i > 0 else 0 for i in range(len(values))]
            mean_return = sum(daily_returns) / len(daily_returns)
            std_dev_return = (sum((r - mean_return)**2 for r in daily_returns) / len(daily_returns))**0.5

            # Calculate the z-score to identify the relative strength of each day's return
            z_scores = [(r - mean_return) / std_dev_return if std_dev_return != 0 else 0 for r in daily_returns]
            max_z_score = max(z_scores)
            seasonal_trends[symbol] = max_z_score

        # Select top_n symbols with the highest seasonality trend
        picks = sorted(seasonal_trends, key=seasonal_trends.get, reverse=True)[:self._top_n]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
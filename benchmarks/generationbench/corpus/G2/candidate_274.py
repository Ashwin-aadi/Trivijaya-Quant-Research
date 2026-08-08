from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrendStrategy(Strategy):
    rationale = (
        "Seasonality effects in equity markets suggest that certain times of the year are more "
        "favorable for particular sectors or stocks. By identifying trends based on historical "
        "performance around specific dates, we can construct a strategy that profits from these "
        "recurring patterns."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        trend_scores = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate the mean close price around each date of interest (e.g., 1st, 15th of the month)
            dates_of_interest = {date(year=stamp.year, month=m, day=d): values[m * 30 + d] for m in range(12) for d in [0, 14]}
            
            # Compute the trend score as the difference between the mean close on date of interest and the overall window mean
            if len(dates_of_interest) > 0:
                mean_close = sum(v for v in dates_of_interest.values()) / len(dates_of_interest)
                trend_scores[symbol] = max([dates_of_interest[date(2020, m + 1, d)] - mean_close for m in range(12) for d in [0, 14]], default=0)

        # Select top performing symbols based on their trend scores
        sorted_symbols = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)
        picks = [s[0] for s in sorted_symbols[:5]]

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
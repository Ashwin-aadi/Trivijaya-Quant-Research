from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityEffect(Strategy):
    rationale = (
        "Historical data often shows that certain stocks exhibit higher returns during specific "
        "times of the year. By identifying these patterns, we can exploit them for trading gains."
    )

    def __init__(self, window: int = 365) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter out non-trading days and focus on the most recent window
        trading_days = [d for d in history["session_date"].to_list() if d < stamp]
        if len(trading_days) < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Group by symbols and calculate mean close price during the window
        grouped_history = (
            history.group_by("symbol").agg(pl.col("adj_close").mean().alias("mean_close"))
        )
        
        # Identify symbols that have higher closes in the recent period compared to their historical average
        breakout_symbols = [
            symbol for symbol, row in grouped_history.iter_rows()
            if float(row["mean_close"]) < view.latest_close()[symbol] and
               (view.history(lookback=self._window)["adj_close"][0][symbol] >= 1.05 * float(row["mean_close"]))
        ]

        # Ensure we do not exceed the number of symbols to consider
        breakout_symbols = breakout_symbols[:3]

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Historical data suggests that certain months of the year exhibit higher returns for "
        "certain stocks due to seasonal patterns. For instance, some industries may benefit from "
        "holiday periods, while others might see increased demand during certain seasons."
    )

    def __init__(self, lookback_period: int = 365, threshold_high: float = 0.15, 
                 threshold_low: float = -0.15) -> None:
        self._lookback_period = lookback_period
        self._threshold_high = threshold_high
        self._threshold_low = threshold_low

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter by session_date to get the current month's returns
        current_month_history = history.filter(
            (pl.col("session_date").dt.month() == stamp.month) &
            (pl.col("session_date") < stamp)
        )

        if current_month_history.height < 20:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the average return for each symbol in the current month
        avg_returns = current_month_history.groupby("symbol").agg(
            pl.col("adj_close").first().alias("open"), 
            (pl.col("adj_close") / pl.col("open") - 1.0).alias("return")
        ).sort("return", descending=True)

        # Filter symbols based on the return threshold
        high_return_symbols = avg_returns.filter(
            (avg_returns["return"] > self._threshold_high)
        ).select(["symbol"]).to_dict(as_series=False)["symbol"]
        
        low_return_symbols = avg_returns.filter(
            (avg_returns["return"] < self._threshold_low)
        ).select(["symbol"]).to_dict(as_series=False)["symbol"]

        # Combine both lists and deduplicate
        selected_symbols = list(set(high_return_symbols + low_return_symbols))

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
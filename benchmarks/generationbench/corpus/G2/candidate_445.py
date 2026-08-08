from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion is a phenomenon where asset prices tend to return to the mean over time. "
        "In the short term, extreme prices are likely to revert towards their historical average."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the mean and standard deviation of the 20-day closing prices
        means = (
            history.group_by("symbol")
            .agg(
                pl.col("adj_close").mean().alias("mean"),
                (pl.col("adj_close") - pl.col("adj_close").mean()).std().alias("std_dev"),
            )
        )

        # Filter for symbols with high deviation from the mean
        filtered = means.filter(
            ((pl.col("adj_close") - pl.col("mean")) / pl.col("std_dev") < -2)
        ).select("symbol")

        if not filtered.height:
            return Signal(information_available_at=stamp, weights={})

        # Select top 5 symbols with the highest negative deviation
        picks: list[str] = [str(s) for s in filtered["symbol"].to_list()][:5]

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
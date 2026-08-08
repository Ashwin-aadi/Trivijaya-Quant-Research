from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Certain stocks in the NIFTY 100 may exhibit predictable patterns of performance based "
        "on seasonal calendar effects. For instance, consumer discretionary sectors might show "
        "strong returns during festive periods like Diwali or Christmas. By identifying such "
        "seasonal trends, we can opportunistically allocate capital."
    )

    def __init__(self, window: int = 365, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._lookback)

        seasonal_factors: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._lookback:
                continue

            # Calculate the mean close during festive periods (Diwali, Christmas)
            diwali_close_mean = history.filter(
                (pl.col("session_date").dt.month() == 10) & (pl.col("session_date").dt.day() >= 22)
            )["adj_close"][symbol].mean()
            christmas_close_mean = history.filter(
                (pl.col("session_date").dt.month() == 12) & (pl.col("session_date").dt.day() == 25)
            )["adj_close"][symbol].mean()

            if not diwali_close_mean.is_nan() and not christmas_close_mean.is_nan():
                seasonal_factors[symbol] = (
                    max(diwali_close_mean, christmas_close_mean) / min(diwali_close_mean, christmas_close_mean)
                )

        sorted_symbols = [
            symbol for symbol in view.symbols if symbol in seasonal_factors
            if seasonal_factors[symbol] > 1.05  # Only consider factors greater than 1.05
        ]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
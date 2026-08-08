from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Seasonality252(Strategy):
    rationale = (
        "Seasonal patterns in stock markets can be exploited by identifying "
        "and capitalizing on historical trends. This strategy focuses on the last 252 trading days to capture annual seasonal effects."
    )

    def __init__(self, window: int = 252, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_weights: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate the mean close over the past 252 days
            mean_close = sum(values[-self._window:]) / self._window

            # Assign weights based on the closeness to the mean
            if abs(values[-1] - mean_close) <= (mean_close * 0.05):
                seasonal_weights[symbol] = 1.0 / self._top_n

        if not seasonal_weights:
            return Signal(information_available_at=stamp, weights={})

        weight = sum(seasonal_weights.values()) / len(seasonal_weights)
        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in seasonal_weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
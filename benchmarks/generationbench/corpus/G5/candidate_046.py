from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Historical data often shows that certain stocks exhibit seasonality effects. "
        "By identifying and exploiting these patterns, we can potentially generate alpha."
    )

    def __init__(self, window: int = 365, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonality_factors: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.unique().to_list():
                continue
            close_series = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()]
            if len(close_series) < self._window:
                continue

            # Calculate the mean closing price over the last window days
            recent_mean = sum(close_series[-self._window:]) / self._window
            current_close = view.latest_close()[symbol]
            factor = (current_close - recent_mean) / recent_mean
            if abs(factor) >= self._threshold:
                seasonality_factors[symbol] = factor

        selected_symbols = [s for s, f in seasonality_factors.items() if f > 0]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest
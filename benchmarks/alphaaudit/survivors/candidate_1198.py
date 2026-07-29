from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality in the Indian market suggests that certain times of the year exhibit "
        "repeated patterns. By identifying these trends, we can capitalize on periodic "
        "market movements."
    )

    def __init__(self, window: int = 260) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_strengths: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            mean_close = sum(values) / self._window
            strength = abs(max(values) - min(values)) / (2 * mean_close)
            seasonal_strengths[symbol] = strength

        strongest_tickers = sorted(seasonal_strengths.items(), key=lambda x: x[1], reverse=True)[:5]
        if not strongest_tickers:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(strongest_tickers)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in strongest_tickers},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
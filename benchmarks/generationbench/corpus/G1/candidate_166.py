from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityEffect(Strategy):
    rationale = (
        "Seasonal effects can be exploited by identifying stocks that exhibit higher returns "
        "at certain times of the year. This strategy aims to capture these anomalies by "
        "allocating capital to stocks with historically strong performance during their peak "
        "seasons."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_signals = {}
        for symbol in view.symbols:
            hist = history.filter(pl.col("symbol") == symbol).select(
                pl.col("session_date"), pl.col("adj_close")
            )
            if hist.height < self._window:
                continue
            closes = [float(v) for v in hist["adj_close"].to_list()]
            peak_month = max(enumerate(closes), key=lambda x: x[1])[0] + 1

            # Assign a score based on the current month relative to the peak month
            current_month = stamp.month
            if current_month == peak_month or (current_month - peak_month) % 12 < 3:
                seasonal_signals[symbol] = (peak_month, closes[-1])

        sorted_symbols = sorted(seasonal_signals.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in sorted_symbols][:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are a common technical trading strategy. "
        "High volume on the direction of a recent trend is indicative of strong momentum and"
        " can lead to continuation of that trend."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            # Calculate the close and volume change
            closes = [float(v) for v in history[symbol].filter(pl.col("session_date") > date(2020, 1, 1))["close"].to_list()]
            volumes = [int(v) for v in history[symbol].filter(pl.col("session_date") > date(2020, 1, 1))["volume"].to_list()]

            if len(closes) < self._window:
                continue

            # Calculate the trend direction
            trend_direction = (closes[-1] - closes[0]) / abs(closes[0])

            # Volume on the last day of window
            latest_volume = volumes[-1]

            # Check for high volume and positive trend
            if latest_volume > 2 * max(volumes) and trend_direction > 0:
                signals[symbol] = 1.0

        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
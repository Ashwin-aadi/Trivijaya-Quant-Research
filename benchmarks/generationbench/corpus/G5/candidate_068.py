from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest strong market sentiment. "
        "By focusing on both price and volume, we can identify potentially significant trends."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = []
        for symbol in view.symbols:
            hist = history.filter(pl.col("symbol") == symbol).sort(by="session_date")
            open_values = [float(v) for v in hist["open"].to_list()]
            close_values = [float(v) for v in hist["close"].to_list()]

            if len(open_values) < self._window:
                continue

            last_open, last_close = open_values[-1], close_values[-1]
            penultimate_open, penultimate_close = (
                open_values[-2],
                close_values[-2],
            )

            # Check for a strong directional move
            direction = None
            if (last_close > penultimate_close and last_open >= penultimate_open) or \
               (last_close < penultimate_close and last_open <= penultimate_open):
                direction = "up" if last_close > penultimate_close else "down"

            # Confirm with volume
            volumes = [float(v) for v in hist["volume"].to_list()]
            recent_volume_avg = sum(volumes[-2:]) / 2  # Average of last two sessions

            if direction and recent_volume_avg > max(volumes[:-1]):
                symbols.append(symbol)

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
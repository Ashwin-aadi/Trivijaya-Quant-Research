from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest strong market sentiment and can indicate "
        "potentially significant trend reversals or continuations. By focusing on both price "
        "and volume, we aim to capture these opportunities more reliably."
    )

    def __init__(self, window: int = 20, volume_multiplier: float = 1.5) -> None:
        self._window = window
        self._volume_multiplier = volume_multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            prices = [float(v) for v in df["adj_close"].to_list()]
            volumes = [float(v) for v in df["volume"].to_list()]

            if len(prices) < self._window or len(volumes) < self._window:
                continue

            direction = 1.0
            if prices[-1] < prices[0]:
                direction = -1.0

            volume_change = volumes[-1] / volumes[0]
            threshold = 1.0 + (self._volume_multiplier * (direction * 0.5))

            if volume_change >= threshold:
                signals[symbol] = direction

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_direction = sum(signals.values())
        weight_per_symbol = total_direction / len(signals)
        weights = {symbol: weight_per_symbol for symbol in signals}

        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
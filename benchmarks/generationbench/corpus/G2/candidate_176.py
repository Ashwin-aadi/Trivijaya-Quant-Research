from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeBreakout(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest that a significant flow of buying or selling "
        "is present. If this volume is accompanied by a price move in the direction of the flow, it "
        "may indicate a continuation or acceleration of an existing trend."
    )

    def __init__(self, window: int = 10, breakout_threshold: float = 1.05) -> None:
        self._window = window
        self._breakout_threshold = breakout_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns or symbol not in closes.columns:
                continue

            close_values = [float(v) for v in history[symbol]["close"].to_list()]
            volume_values = [float(v) for v in history[symbol]["volume"].to_list()]

            latest_close = float(closes[0][symbol])
            previous_close = close_values[-1]
            breakout_condition = (latest_close - previous_close) / previous_close >= self._breakout_threshold

            if not breakout_condition:
                continue

            # Check if the volume has increased significantly
            last_volume = volume_values[-1]
            volume_change = max(volume_values) / last_volume > 2.0

            if volume_change:
                breakout_signals[symbol] = 1.0

        symbols_with_breakout = sorted(breakout_signals, key=breakout_signals.get, reverse=True)
        weight_per_symbol = 1.0 / len(symbols_with_breakout) if symbols_with_breakout else 0
        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_symbol for s in symbols_with_breakout},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
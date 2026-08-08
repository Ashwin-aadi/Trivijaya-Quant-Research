from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong buying or selling pressure. "
        "By identifying such moves, we can capitalize on the continuation of trends."
    )

    def __init__(self, window: int = 10, threshold: float = 0.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            symbol_history = history.filter(pl.col("symbol") == symbol).sort("session_date")
            open_values = [float(v) for v in symbol_history["open"].to_list()]
            close_values = [float(v) for v in symbol_history["close"].to_list()]
            volume_values = [float(v) for v in symbol_history["volume"].to_list()]

            if len(open_values) < self._window:
                continue

            direction = 1.0 if close_values[-1] > open_values[0] else -1.0
            avg_volume = sum(volume_values) / self._window

            # Volume confirmation check
            max_volume = max(volume_values)
            volume_confirm = max_volume >= self._threshold * avg_volume

            if direction == 1 and volume_confirm:
                signals[symbol] = direction
            elif direction == -1 and volume_confirm:
                signals[symbol] = direction

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_directions = sum(signals.values())
        weight_per_symbol = 1.0 / len(signals) * abs(total_directions)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight_per_symbol if direction > 0 else -weight_per_symbol
                for symbol, direction in signals.items()
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
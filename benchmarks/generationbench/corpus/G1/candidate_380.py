from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are often seen as a sign of strong market sentiment "
        "and can indicate potential continuation of the trend. By identifying such moves, we aim "
        "to capture profitable trades while minimizing false signals."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_series = [float(v) for v in history[symbol]["adj_close"].to_list()]
            volume_series = [int(v) for v in history[symbol]["volume"].to_list()]

            if len(close_series) < self._window or len(volume_series) < self._window:
                continue

            last_close = close_series[-1]
            mean_close = sum(close_series[-self._window:]) / self._window
            last_volume = volume_series[-1]
            mean_volume = sum(volume_series[-self._window:]) / self._window

            if last_close > mean_close and last_volume > 1.5 * mean_volume:
                signals[symbol] = 1.0
            elif last_close < mean_close and last_volume > 1.5 * mean_volume:
                signals[symbol] = -1.0

        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, weight in signals.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
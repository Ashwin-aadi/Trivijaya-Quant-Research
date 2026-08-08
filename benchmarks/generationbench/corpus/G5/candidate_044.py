from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmDirectionalMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves identify significant shifts in market sentiment "
        "by combining the magnitude of price changes with their accompanying volume. High volume "
        "on a strong price move is often indicative of genuine momentum and can lead to sustained "
        "price action."
    )

    def __init__(self, window: int = 20, volume_threshold: float = 1.2) -> None:
        self._window = window
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_confirmed_signals: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            close_changes = [float(v2 / v1 - 1.0) for v1, v2 in zip(history[f"close_{symbol}"].shift(1).to_list(), history[f"close_{symbol}"].to_list())[1:]]
            volume_changes = [float(v2 / v1) for v1, v2 in zip(history[f"volume_{symbol}"].shift(1).to_list(), history[f"volume_{symbol}"].to_list())[1:]]

            for i in range(len(close_changes)):
                if close_changes[i] > 0 and volume_changes[i] > self._volume_threshold:
                    volume_confirmed_signals[symbol] = (close_changes[i] + 1) / self._window
                elif close_changes[i] < 0 and volume_changes[i] > self._volume_threshold:
                    volume_confirmed_signals[symbol] = -(close_changes[i] + 1) / self._window

        if not volume_confirmed_signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(volume_confirmed_signals.values())
        weighted_signals = {symbol: weight / total_weight for symbol, weight in volume_confirmed_signals.items()}

        return Signal(
            information_available_at=stamp,
            weights=weighted_signals
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
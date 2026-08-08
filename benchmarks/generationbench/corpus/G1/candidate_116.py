from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate a strong market direction and can "
        "potentially lead to sustained price movements. By identifying such moves, we aim "
        "to capitalize on the trend with confidence."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in symbols:
            close_values = history[symbol].to_list()
            volume_values = history[f"{symbol}_volume"].to_list()

            if len(close_values) < self._window + 1 or len(volume_values) < self._window + 1:
                continue

            # Calculate daily returns
            returns = [(close_values[i] - close_values[i-1]) / close_values[i-1] for i in range(1, self._window + 1)]
            volumes = [volume_values[i] for i in range(self._window)]

            # Check for volume-confirmed directional move
            if all(volumes[i] > volumes[0] for i in range(1, self._window)):
                signals[symbol] = (sum(returns) / len(returns)) * 0.95

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_signal_strength = sum(signals.values())
        weights = {symbol: signal / total_signal_strength for symbol, signal in signals.items()}
        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items() if w > 0.01}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
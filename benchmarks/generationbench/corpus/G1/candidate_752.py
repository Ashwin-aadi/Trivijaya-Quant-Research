from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves can signal sustained price trends. "
        "By identifying symbols with both significant price changes and corresponding volume increases, "
        "we can potentially capture profitable trends before they reverse."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_changes = (
                history[f"{symbol}_close"]
                / history[f"{symbol}_close"].shift(1) - 1.0
            ).to_list()
            volume_changes = (
                history[f"{symbol}_volume"]
                / history[f"{symbol}_volume"].shift(1)
            ).to_list()

            if len(close_changes) < self._window:
                continue

            close_change = close_changes[-1]
            volume_change = volume_changes[-1]

            if abs(close_change) > self._threshold and volume_change > 1.0:
                signals[symbol] = close_change / (len(view.symbols) + 1)

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(signals.values())
        normalized_weights = {s: w / total_weight for s, w in signals.items()}
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, weight in normalized_weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
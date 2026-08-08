from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves can indicate strong buying or selling pressure. "
        "High volume on a price breakout may signal that the move is not just a temporary blip but a more sustained trend."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"]:
                continue
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            closes = [float(v) for v in df["adj_close"].to_list()]
            volumes = [int(v) for v in df["volume"].to_list()]

            if len(closes) < self._window:
                continue

            latest_close = closes[-1]
            breakout_threshold = max(closes[:-1]) + (max(closes[:-1]) - min(closes[:-1])) / 4
            if latest_close >= breakouthreshold and volumes[-1] > sum(volumes[-self._window : -1]):
                signals[symbol] = 0.5

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(signals.values())
        scaled_weights = {s: w / total_weight for s, w in signals.items()}
        return Signal(
            information_available_at=stamp,
            weights=scaled_weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
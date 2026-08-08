from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are signals that a stock is experiencing strong "
        "market interest and could continue in the same direction. When volume spikes on high "
        "price movement, it suggests that large traders or institutions are participating, which "
        "may indicate a continuation of the trend."
    )

    def __init__(self, window: int = 20, threshold_ratio: float = 1.5) -> None:
        self._window = window
        self._threshold_ratio = threshold_ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_change = (history["high"] / history["close"].shift(1) - 1.0).alias("high_change")
        low_change = (history["low"] / history["close"].shift(1) - 1.0).alias("low_change")
        volume_change = (history["volume"] / history["volume"].shift(1)).alias("volume_change")

        high_slope = (pl.col("high").rank(method="dense", descending=True).to_list()[-1] -
                      pl.col("high").rank(method="dense", descending=True).to_list()[0]) / self._window
        low_slope = (pl.col("low").rank(method="dense", descending=True).to_list()[-1] -
                     pl.col("low").rank(method="dense", descending=True).to_list()[0]) / self._window

        history = (
            history
            .with_columns(high_change, low_change, volume_change)
            .sort("session_date", descending=False)
        )

        high_changes = [float(v) for v in history["high_change"].drop_nulls().to_list()]
        low_changes = [float(v) for v in history["low_change"].drop_nulls().to_list()]
        volumes = [float(v) for v in history["volume_change"].drop_nulls().to_list()]

        breakout_suspects: list[str] = []

        for symbol in view.symbols:
            if (
                symbol not in high_changes
                or symbol not in low_changes
                or symbol not in volumes
            ):
                continue

            if max(high_changes) >= self._threshold_ratio and max(volumes):
                breakout_suspects.append(symbol)
            elif min(low_changes) <= -self._threshold_ratio and max(volumes):
                breakout_suspects.append(symbol)

        weights = {s: 1.0 / len(breakout_suspects) for s in breakout_suspects}
        return Signal(
            information_available_at=stamp, weights={**weights}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
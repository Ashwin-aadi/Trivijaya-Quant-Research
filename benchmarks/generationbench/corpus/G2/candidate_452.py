from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMove(Strategy):
    rationale = (
        "High volume breakout signals a significant change in investor sentiment. "
        "If the market breaks out of its recent range on high volume, it may indicate "
        "that the price is likely to continue moving in that direction."
    )

    def __init__(self, window: int = 20, volume_threshold: float = 1.5) -> None:
        self._window = window
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or not all(symbol in history.columns for symbol in view.symbols):
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            row = history.filter(pl.col("symbol") == symbol).sort("session_date")
            adj_closes = [float(v) for v in row["adj_close"].to_list()]
            volumes = [float(v) for v in row["volume"].to_list()]

            if len(adj_closes) < self._window:
                continue

            # Calculate the relative volume change
            last_volume = volumes[-1]
            mean_volume = sum(volumes) / len(volumes)
            relative_volume_change = (last_volume - mean_volume) / mean_volume

            # Check for breakout on high volume
            if adj_closes[-1] >= max(adj_closes) and relative_volume_change > self._volume_threshold:
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
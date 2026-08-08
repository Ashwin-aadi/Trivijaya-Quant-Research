from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are more likely to continue than those that "
        "occur without strong volume support. This strategy seeks to capture such moves by"
        " identifying significant price increases with substantial volume."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            df = history.select(["session_date", "open", "close", "volume"])
            open_vals = [float(v) for v in df["open"].to_list()]
            close_vals = [float(v) for v in df["close"].to_list()]
            volume_vals = [float(v) for v in df["volume"].to_list()]

            if len(open_vals) < self._window:
                continue

            # Calculate directional move
            direction = (close_vals[-1] - open_vals[0]) / open_vals[0]
            # Calculate volume change from first to last session
            volume_change = volume_vals[-1] / volume_vals[0]

            if direction > 0.05 and volume_change >= 1.2:  # Adjusted parameters
                picks[symbol] = (direction + volume_change) * 0.5

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(picks.values())
        weight_per_symbol = {s: w / total_weight for s, w in picks.items() if w > 0}
        return Signal(
            information_available_at=stamp,
            weights=weight_per_symbol,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
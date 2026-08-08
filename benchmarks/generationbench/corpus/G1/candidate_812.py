from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong momentum in a stock. "
        "These moves can be used to identify potential breakout or trend continuation opportunities."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            volume_values = [
                float(v)
                for v in history[f"{symbol}_volume"].drop_nulls().to_list()
            ]
            if len(close_values) < self._window:
                continue

            # Calculate directional change
            direction = (close_values[-1] - close_values[0]) / close_values[0]
            volume_change = (volume_values[-1] - volume_values[0]) / volume_values[0]

            # Filter for significant directional move and high volume confirmation
            if abs(direction) > 0.05 and abs(volume_change) > 0.2:
                picks.append(symbol)

        weights = {s: 1.0 / len(picks) for s in picks} if picks else {}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().date()
    assert isinstance(newest, date)
    return newest
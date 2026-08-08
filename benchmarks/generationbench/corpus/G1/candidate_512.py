from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are likely to continue in the near future. "
        "This strategy identifies symbols that have shown strong momentum over a short period, "
        "backed by increased trading volume."
    )

    def __init__(self, window: int = 10, threshold_multiplier: float = 2.0) -> None:
        self._window = window
        self._threshold_multiplier = threshold_multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue

            close_changes = [float(v) for v in (history[f"close_lag_{i+1}"] - history["adj_close"]).to_list()]
            volume_changes = [float(v) for v in (history[f"volume_lag_{i+1}"] - history["volume"]).to_list()]

            if len(close_changes) < self._window:
                continue

            # Calculate the directional move and its strength
            directionality = sum([c > 0 for c in close_changes[-self._window:]])
            volume_strength = max(volume_changes)[-self._window:] / min(volume_changes)[-self._window:]

            if directionality >= self._threshold_multiplier:
                picks.append(symbol)

        picks = picks[:5]  # Limit to top 5 symbols
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
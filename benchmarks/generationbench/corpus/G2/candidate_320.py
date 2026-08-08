from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "High volume days often indicate significant buying or selling pressure. "
        "A directional move (up or down) accompanied by high volume can signal a strong trend "
        "and potential continuation of the movement."
    )

    def __init__(self, window: int = 5, threshold: float = 1.2) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        moves: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns or "volume" not in history.columns:
                continue
            close_values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            volume_values = [float(v) for v in history["volume"].drop_nulls().to_list()]
            if len(close_values) < self._window or len(volume_values) < self._window:
                continue

            last_close = close_values[-1]
            for i in range(self._window - 1, 0, -1):
                if (close_values[i] > close_values[i - 1]) == (last_close > close_values[-1]):
                    # Check for direction consistency
                    if volume_values[i] / volume_values[i - 1] >= self._threshold:
                        moves[symbol] = last_close
                        break

        if not moves:
            return Signal(information_available_at=stamp, weights={})

        weights = {symbol: 1.0 / len(moves) for symbol in moves}
        return Signal(
            information_available_at=stamp,
            weights={s: weights[s] for s in view.symbols if s in weights},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
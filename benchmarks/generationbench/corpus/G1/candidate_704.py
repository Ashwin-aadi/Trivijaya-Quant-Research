from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves can indicate strong buying or selling pressure. "
        "By combining price action with volume, we can identify potential trend reversals or continuation."
    )

    def __init__(self, window: int = 20, confirm_window: int = 5) -> None:
        self._window = window
        self._confirm_window = confirm_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._confirm_window)
        if history.height < self._window + self._confirm_window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            open_price = [float(v) for v in history[symbol]["open"].to_list()]
            close_price = [float(v) for v in history[symbol]["close"].to_list()]
            volume = [int(v) for v in history[symbol]["volume"].to_list()]

            if len(open_price) < self._window + self._confirm_window:
                continue

            last_open = open_price[-1]
            last_close = close_price[-1]

            # Calculate directional move
            direction = 1.0 if last_close > last_open else -1.0

            # Confirm with volume on the next window
            confirm_volume = sum(volume[-self._confirm_window:])
            total_volume = sum(volume)

            if (direction == 1 and confirm_volume / total_volume > 0.5) or \
                    (direction == -1 and confirm_volume / total_volume < 0.3):
                signals[symbol] = direction

        # Normalize weights
        weight_sum = sum(signals.values())
        normalized_weights = {symbol: value / weight_sum for symbol, value in signals.items()}
        
        if not signals:
            return Signal(information_available_at=stamp, weights={})
        
        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in normalized_weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
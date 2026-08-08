from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirm(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of strong market sentiment and "
        "greater likelihood of continued movement in the direction of the initial move."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            symbol_history = history[symbol]
            recent_close = float(symbol_history["close"].max())
            recent_volume = int(symbol_history["volume"].max())

            # Check for a directional move with sufficient volume confirmation
            moves = [float(v) for v in symbol_history["close"].to_list()]
            if len(moves) < self._window:
                continue

            latest_move = recent_close - float(symbol_history["close"][1])
            if not (latest_move > 0 and recent_volume >= max([int(volume) for volume in symbol_history["volume"]])):
                continue

            # Check the previous window to ensure it was a downward move
            prev_window_moves = [moves[i] - moves[i + 1] for i in range(len(moves) - 1)]
            if not all(move < 0 for move in prev_window_moves):
                continue

            breakout_symbols.append(symbol)

        weights: dict[str, float] = {symbol: 1.0 / len(breakout_symbols) for symbol in breakout_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
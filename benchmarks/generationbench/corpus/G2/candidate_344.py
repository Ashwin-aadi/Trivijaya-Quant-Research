from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest that large institutional traders are driving "
        "the market. Such moves can be a leading indicator of sustained price action in the future."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_confirmed_moves = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            adj_closes = [float(v) for v in history[symbol].to_list()]
            volumes = [float(v) for v in history["volume"].to_list()]

            # Calculate the price change over the window
            recent_close = adj_closes[-1]
            recent_volume = volumes[-1]
            previous_window_close = sum(adj_closes[:-1]) / (self._window - 1)

            if not all(volumes):
                continue

            # Check for a strong volume day and a directional move
            if recent_volume > max(volumes) * 0.75 and abs(recent_close - previous_window_close) >= recent_close * 0.02:
                volume_confirmed_moves[symbol] = (recent_close, previous_window_close)

        # Identify symbols with the strongest moves
        sorted_moves = sorted(volume_confirmed_moves.items(), key=lambda x: abs(x[1][0] - x[1][1]), reverse=True)
        picks = [symbol for symbol, _ in sorted_moves[:5]]

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
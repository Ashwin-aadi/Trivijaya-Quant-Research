from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of significant market interest and "
        "can lead to sustained trends. By identifying such moves in the NIFTY 100 constituents, "
        "we can capitalize on these moments of heightened trading activity."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_confirmed_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            daily_volumes = [float(v) for v in history[symbol + "_volume"].drop_nulls().to_list()]
            adj_closes = [float(v) for v in history[symbol + "_adj_close"].drop_nulls().to_list()]

            if len(daily_volumes) < self._window or len(adj_closes) < self._window:
                continue

            # Calculate the directional move
            direction = (adj_closes[-1] - adj_closes[0]) / adj_closes[0]
            avg_volume = sum(daily_volumes) / self._window

            if direction > 0.05 and daily_volumes[-1] > 1.2 * avg_volume:
                volume_confirmed_symbols.append(symbol)

        if not volume_confirmed_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(volume_confirmed_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in volume_confirmed_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
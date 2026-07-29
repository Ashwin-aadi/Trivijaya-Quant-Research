from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMoves(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of strong buying or selling pressure. "
        "This strategy captures such moves by identifying symbols with significant volume increases "
        "on days where the price also shows a substantial change in direction."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols_with_moves = []
        for symbol in view.symbols:
            symbol_history = history.filter(pl.col("symbol") == symbol).sort(
                "session_date"
            )
            if symbol_history.height < self._window:
                continue

            last_close = float(symbol_history["adj_close"].last())
            first_close = float(symbol_history["adj_close"][0])
            volume_changes = [float(v) for v in symbol_history["volume"].to_list()]
            price_change = (last_close - first_close) / first_close
            if abs(price_change) > 0.05 and max(volume_changes) >= sum(volume_changes) * 0.1:
                symbols_with_moves.append(symbol)

        if not symbols_with_moves:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_with_moves)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in symbols_with_moves}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
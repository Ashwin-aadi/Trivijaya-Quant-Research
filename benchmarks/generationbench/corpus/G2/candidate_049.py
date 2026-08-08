from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion5d(Strategy):
    rationale = (
        "Mean reversion occurs when asset prices tend to revert to a historical mean. "
        "Short-horizon mean reversion strategies look for extreme price movements in the past 5 days and "
        "bet on a return towards the average price level."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date").tail(self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = float(history["adj_close"].mean())
        symbols_with_extreme_moves: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            recent_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if abs(recent_closes[-1] - mean_close) >= 0.2 * mean_close:
                symbols_with_extreme_moves.append(symbol)

        if not symbols_with_extreme_moves:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_with_extreme_moves)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in symbols_with_extreme_moves}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
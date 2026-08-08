from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts are often followed by continuation of the breakout direction. Identifying "
        "and capitalizing on such continuations can generate profits."
    )

    def __init__(self, window: int = 30, threshold: float = 0.05) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            high_close_ratio = (history[f"{symbol}_high"] / history[f"{symbol}_close"]).to_list()[-1]
            low_close_ratio = (history[f"{symbol}_low"] / history[f"{symbol}_close"]).to_list()[0]

            # Check for breakout condition
            if high_close_ratio > 1 + self._threshold:
                breakout_symbols.append(symbol)
            elif low_close_ratio < 1 - self._threshold:
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Identify continuation of breakout direction
        continuation_weights: dict[str, float] = {}
        for symbol in breakout_symbols:
            history = view.history(lookback=self._window)
            direction = (history[f"{symbol}_close"] / history[f"{symbol}_close"].shift(self._window) - 1).to_list()[-2]
            if direction > self._threshold or direction < -self._threshold:
                continuation_weights[symbol] = 0.5
            else:
                continuation_weights[symbol] = 0

        total_weight = sum(continuation_weights.values())
        weights = {s: (w / total_weight) for s, w in continuation_weights.items() if w != 0}
        return Signal(
            information_available_at=stamp,
            weights={**weights, **{symbol: 1 - total_weight for symbol in weights}}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
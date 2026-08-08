from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in the direction they broke out. By identifying stocks that "
        "have already shown some strength after a breakout and increasing our position size, we "
        "can potentially capture more of the post-breakout momentum."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].to_list()]
            high_close_ratio = max(prices[-20:]) / prices[-1]
            if high_close_ratio > 1.05:  # Adjust the threshold as needed
                breakout_symbols.append(symbol)

        continuation_symbols = []
        for symbol in breakout_symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].to_list()]
            if len(prices) < self._window:
                continue
            last_20_days_high = max(prices[-self._window:])
            last_day_close = prices[-1]
            if (last_day_close - last_20_days_high) / last_20_days_high > 0.05:  # Adjust the threshold as needed
                continuation_symbols.append(symbol)

        continuation_symbols = continuation_symbols[: self._top_n]
        if not continuation_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(continuation_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in continuation_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
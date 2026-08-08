from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a strong breakout, the market often consolidates and then continues in the "
        "same direction. This strategy looks for symbols that have recently broken out and "
        "are now trading near their 20-day high, suggesting a continuation pattern."
    )

    def __init__(self, window: int = 20, threshold: float = 0.95) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_values = [float(v) for v in history[symbol].to_list()]
            high_value = max(close_values[-self._window:])
            last_close = close_values[-1]
            if last_close > (high_value * self._threshold):
                breakout_symbols.append(symbol)

        breakout_symbols = list(set(breakout_symbols))  # Remove duplicates
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
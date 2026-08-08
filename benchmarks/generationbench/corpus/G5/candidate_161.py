from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After identifying a breakout in the previous 20 days, this strategy looks for "
        "confirmation on the next day to enter a position. This is based on the idea that "
        "a strong move one day may continue the next."
    )

    def __init__(self, window: int = 20, confirmation_threshold: float = 0.05) -> None:
        self._window = window
        self._confirmation_threshold = confirmation_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(values) < self._window + 1:
                continue
            if values[-1] > max(values[:self._window]):
                breakout_symbols.append(symbol)

        continuation_symbols = []
        for symbol in breakout_symbols:
            next_day_close = view.latest_close()[symbol]
            if (next_day_close - history[symbol][-2]) / history[symbol][-2] > self._confirmation_threshold:
                continuation_symbols.append(symbol)

        weights = {s: 1.0 / len(continuation_symbols) for s in continuation_symbols}
        return Signal(
            information_available_at=stamp, weights={**weights}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
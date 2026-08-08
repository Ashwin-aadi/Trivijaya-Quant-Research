from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Breakouts that are sustained by further price movements in the breakout direction "
        "indicate a strong trend continuation. This strategy identifies symbols that have "
        "recently broken out and continue to move in that direction."
    )

    def __init__(self, window: int = 20, continuation_window: int = 5, min_cont_factor: float = 0.1) -> None:
        self._window = window
        self._continuation_window = continuation_window
        self._min_cont_factor = min_cont_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_window)
        if history.height < self._window + self._continuation_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = set()
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            breakouts = adj_closes[-self._window - 1 : -self._continuation_window]
            continuations = adj_closes[-self._continuation_window:]
            if len(breakouts) < self._window or len(continuations) < self._continuation_window:
                continue

            breakout_price = breakouts[0]
            max_continuation_price = max(continuations)
            cont_factor = (max_continuation_price - breakout_price) / breakout_price
            if cont_factor > self._min_cont_factor:
                breakout_symbols.add(symbol)

        weights = {s: 1.0 / len(breakout_symbols) for s in breakout_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
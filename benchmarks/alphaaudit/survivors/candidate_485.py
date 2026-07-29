from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Continuation breakouts occur after an initial breakout. If a stock breaks out to "
        "new highs but then retraces slightly before continuing the move, it signals potential"
        " continuation of the trend."
    )

    def __init__(self, initial_window: int = 20, retrace_window: int = 10) -> None:
        self._initial_window = initial_window
        self._retrace_window = retrace_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._initial_window + self._retrace_window)

        if history.height < self._initial_window + self._retrace_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            symbol_history = history.filter(pl.col("symbol") == symbol)
            initial_highs = [float(v) for v in symbol_history["high"].drop_nulls().to_list()][-self._initial_window:]
            retrace_highs = [float(v) for v in symbol_history["high"].drop_nulls().to_list()][-self._retrace_window:]

            if len(initial_highs) < self._initial_window or len(retrace_highs) < self._retrace_window:
                continue

            last_initial_high = max(initial_highs)
            retraced_high = max(retrace_highs)

            if last_initial_high <= retraced_high:
                continue

            breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
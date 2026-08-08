from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Breakouts from long-term ranges can signal a continuation of the prevailing trend. "
        "After identifying symbols that have broken out, we look for confirmation in subsequent "
        "trading sessions to ensure the breakout is sustainable and not just noise."
    )

    def __init__(self, window: int = 20, confirmation_window: int = 5) -> None:
        self._window = window
        self._confirmation_window = confirmation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._confirmation_window)

        if history.height < self._window + self._confirmation_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns or len(history[symbol].to_list()) != self._window:
                continue
            open_values = [float(v) for v in history[symbol]["open"].drop_nulls().to_list()]
            close_values = [float(v) for v in history[symbol]["close"].drop_nulls().to_list()]

            if close_values[-1] >= max(open_values):
                breakout_symbols.append(symbol)

        continuation_symbols = []
        for symbol in breakout_symbols:
            if len(history[symbol].to_list()) < self._window + self._confirmation_window:
                continue
            open_values = [float(v) for v in history[symbol]["open"].drop_nulls().to_list()]
            close_values = [float(v) for v in history[symbol]["close"].drop_nulls().to_list()]

            if any(
                open_values[i] < min(close_values[:i + 1]) and
                close_values[i + self._window - 1] >= max(open_values[i:i + self._confirmation_window])
                for i in range(len(close_values) - self._window)
            ):
                continuation_symbols.append(symbol)

        if not continuation_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(continuation_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in continuation_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
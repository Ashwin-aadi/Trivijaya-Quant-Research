from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in the same direction. "
        "This strategy identifies symbols that have broken out and then continued trending."
    )

    def __init__(self, window: int = 20, continuation_window: int = 10) -> None:
        self._window = window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_window)
        if history.height < self._window + self._continuation_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = _find_breakout_symbols(history)
        continuation_symbols = [
            sym
            for sym in breakout_symbols
            if _check_continuation(
                view.closes(lookback=self._continuation_window), sym
            )
        ]

        if not continuation_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(continuation_symbols)
        return Signal(
            information_available_at=stamp,
            weights={sym: weight for sym in continuation_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _find_breakout_symbols(history: pl.DataFrame) -> list[str]:
    symbols = []
    for symbol in history.columns:
        if len(history[symbol].drop_nulls().to_list()) < 2 * (self._window + self._continuation_window):
            continue

        values = [float(v) for v in history[symbol].drop_nulls().to_list()[-(self._window + self._continuation_window):]]
        if len(values) < self._window + self._continuation_window:
            continue
        last_value = values[0]
        breakout_price = max(values[: self._window])
        if last_value < breakout_price and values[-1] >= breakout_price:
            symbols.append(symbol)
    return symbols


def _check_continuation(closes: pl.DataFrame, symbol: str) -> bool:
    recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
    if len(recent_closes) < self._continuation_window + 1:
        return False
    last_close = recent_closes[-1]
    for close in reversed(recent_closes[:-1]):
        if close >= last_close:
            return False
    return True
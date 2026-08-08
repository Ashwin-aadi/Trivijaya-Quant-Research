from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a breakout, if the price action continues in the breakout direction for some "
        "periods, it suggests a higher likelihood of sustained momentum and potential for "
        "further gains. This strategy captures such continuations by identifying symbols that "
        "have broken out and continue to move in their breakout direction."
    )

    def __init__(self, window: int = 20, continuation_window: int = 5) -> None:
        self._window = window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            last_close = values[-1]
            breakout_price = max(values[: -self._continuation_window])
            if last_close > breakout_price:
                breakout_symbols.append(symbol)

        continuation_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol not in breakout_symbols:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window + self._continuation_window:
                continue
            last_close = values[-1]
            first_continuation_price = min(
                values[self._window : self._window + self._continuation_window]
            )
            if last_close > first_continuation_price:
                continuation_symbols.append(symbol)

        continuation_symbols = list(set(breakout_symbols).intersection(continuation_symbols))
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
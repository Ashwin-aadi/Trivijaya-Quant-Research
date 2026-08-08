from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts that are sustained beyond their initial breakout period suggest continued "
        "momentum. Identifying such breakouts can lead to profitable trades by capturing the "
        "post-breakout momentum."
    )

    def __init__(self, window: int = 20, continuation_window: int = 10) -> None:
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
            if len(values) < self._window + 1:
                continue
            breakout_price = max(values[:-self._continuation_window])
            last_close = values[-1]
            if last_close > breakout_price:
                breakout_symbols.append(symbol)

        continuation_symbols: list[str] = []
        for symbol in breakout_symbols:
            if symbol not in view.history(lookback=self._window).columns:
                continue
            history = view.history(lookback=self._continuation_window + self._window)[symbol]
            values = [float(v) for v in history.drop_nulls().to_list()]
            last_close = values[-1]
            if any(value > last_close for value in values[-self._continuation_window :]):
                continuation_symbols.append(symbol)

        continuation_symbols = list(set(breakout_symbols).intersection(continuation_symbols))
        if not continuation_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(continuation_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in continuation_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
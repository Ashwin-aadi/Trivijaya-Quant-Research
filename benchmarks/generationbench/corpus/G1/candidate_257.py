from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts that continue to rise over a defined period suggest strong momentum and "
        "potential for further gains. By identifying such breakouts early, we can capitalize on "
        "the prevailing trend."
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
            if values[-1] >= max(values):
                breakout_symbols.append(symbol)

        continuation_symbols: list[str] = []
        for symbol in breakout_symbols:
            history = view.history(lookback=self._continuation_window)
            symbol_data = history.filter(pl.col("symbol") == symbol)[["session_date", "close"]]
            if symbol_data.height < self._continuation_window:
                continue
            if all(symbol_data["close"].to_list()[i] > symbol_data["close"].to_list()[i - 1]
                   for i in range(1, self._continuation_window)):
                continuation_symbols.append(symbol)

        weights = {s: 1.0 / len(continuation_symbols) for s in continuation_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
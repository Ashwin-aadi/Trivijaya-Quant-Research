from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "A breakout from a previous range can indicate the start of a new trend. "
        "If the market continues in that direction for a period, it may signal strength and "
        "the continuation of the trend. This strategy identifies symbols that have recently "
        "broken out and continue to move in that direction."
    )

    def __init__(self, breakout_window: int = 20, continuation_window: int = 10) -> None:
        self._breakout_window = breakout_window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._breakout_window + self._continuation_window)

        if history.height < self._breakout_window + self._continuation_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = self._find_breakouts(history)
        continuation_symbols = [
            sym for sym in breakout_symbols if view.closes().get_column(sym).is_monotonically_increasing()
        ]

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


def _find_breakouts(history: pl.DataFrame) -> list[str]:
    symbols = history["symbol"].to_list()
    breakout_symbols = []

    for symbol in symbols:
        symbol_data = history.select(pl.col("symbol") == symbol).drop_nulls().sort("session_date").to_dict()[0][1]
        if len(symbol_data) < 2 * _breakout_window:
            continue
        breakout_price = float(max(symbol_data[-_breakout_window:]))
        for i in range(_breakout_window, len(symbol_data)):
            if symbol_data[i] > breakout_price:
                break
        else:
            continue

        if all(symbol_data[i + 1 : min(i + _continuation_window + 1, len(symbol_data))] > symbol_data[i]
               for i in range(len(symbol_data) - _breakout_window - _continuation_window)):
            breakout_symbols.append(symbol)

    return breakout_symbols
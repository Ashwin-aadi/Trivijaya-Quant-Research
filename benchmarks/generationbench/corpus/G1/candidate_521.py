from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After identifying a breakout, the strategy looks for continuation of that trend. "
        "This is based on the idea that once a price has broken out and is moving in a new direction, "
        "it is likely to continue in that direction."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = set()
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].to_list()]
            if len(values) < self._window + 1:
                continue

            open_price, close_price = float(values[0]), float(values[-1])
            high_low_diff = max(values[1:self._window]) - min(values[1:self._window])

            # Check for breakout
            if (
                (close_price > open_price and close_price >= max(values[1:-1])) or
                (close_price < open_price and close_price <= min(values[1:-1]))
            ):
                breakout_symbols.add(symbol)

        continuation_symbols = set()
        for symbol in breakout_symbols:
            history_symbol = history[symbol]
            if history_symbol.height < self._window + 1:
                continue

            values = [float(v) for v in history_symbol.to_list()]
            open_price, close_price = float(values[0]), float(values[-1])

            # Check for continuation
            if (
                (close_price > open_price and any(close_price >= value for value in values[:-1])) or
                (close_price < open_price and any(close_price <= value for value in values[:-1]))
            ):
                continuation_symbols.add(symbol)

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
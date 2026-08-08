from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Once a breakout has occurred, if the price continues to move in the direction of the "
        "breakout for a certain period, it suggests sustained momentum and increased probability "
        "of further movement in that direction. This strategy identifies such continuation patterns."
    )

    def __init__(self, initial_window: int = 20, continuation_window: int = 5) -> None:
        self._initial_window = initial_window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._initial_window)
        if closes.height < self._initial_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._initial_window:
                continue

            last_close = values[-1]
            max_price = max(values)
            min_price = min(values)

            # Check for breakout condition
            if last_close > max_price or last_close < min_price:
                breakouts = [1.0 if v == max_price else -1.0 if v == min_price else 0.0 for v in values[-self._continuation_window:]]
                continuation_count = sum(breakouts)
                if abs(continuation_count) >= self._continuation_window // 2:
                    breakout_symbols.append(symbol)

        breakout_symbols = list(set(breakout_symbols))[:5]
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
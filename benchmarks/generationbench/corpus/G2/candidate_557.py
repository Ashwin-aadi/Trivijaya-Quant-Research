from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a significant breakout, the continuation of that move is often profitable. "
        "By identifying symbols that have already broken out and are still in an uptrend, we can "
        "capitalize on this momentum."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback)

        if history.is_empty() or history.height < self._window + self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            # Calculate the breakout point
            close_series = [float(v) for v in history[symbol].drop_nulls().to_list()]
            breakout_price = max(close_series[-self._window:])
            breakout_date = _find_breakout_date(close_series, breakout_price)

            # Check if there was a breakout and the price has continued to rise
            if (
                close_series[breakout_date] == breakout_price
                and any(c > breakout_price for c in close_series[(breakout_date + 1) : (breakout_date + self._lookback)])
            ):
                breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[:5]
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols},
        )


def _find_breakout_date(close_series: list[float], breakout_price: float) -> int:
    for i, price in enumerate(reversed(close_series)):
        if price == breakout_price:
            return len(close_series) - 1 - i
    return -1


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
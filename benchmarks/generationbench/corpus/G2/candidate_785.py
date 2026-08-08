from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After an initial breakout, stocks that continue to move in the same direction for a "
        "short period are likely to see further momentum. This strategy identifies such stocks and "
        "allocates capital accordingly."
    )

    def __init__(self, window1: int = 20, window2: int = 5) -> None:
        self._window1 = window1
        self._window2 = window2

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window2 + self._window1)

        if history.is_empty() or history.height < self._window2 + self._window1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(prices) < self._window2 + self._window1:
                continue

            breakout_price = prices[-self._window1]
            breakout_high = max(prices[-self._window1:])
            breakout_low = min(prices[-self._window1:])

            continuation = all(
                price > breakout_price for price in prices[-self._window2:]
            ) or all(price < breakout_price for price in prices[-self._window2:])

            if (
                prices.index(breakout_high) >= len(prices) - self._window1
                and prices.index(breakout_low) <= len(prices) - self._window1
                and continuation
            ):
                breakout_symbols.append(symbol)

        breakout_symbols = list(set(breakout_symbols))
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts are often followed by price continuation. Identifying stocks that have "
        "recently broken out and showing strong momentum can lead to profitable trades."
    )

    def __init__(self, window1: int = 20, window2: int = 5) -> None:
        self._window1 = window1
        self._window2 = window2

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window1 + self._window2)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = set()
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].to_list()]
            if len(prices) < self._window1 + self._window2:
                continue

            close_prices = prices[-self._window1 :]
            breakout_price = max(close_prices)
            breakout_date = history["session_date"][close_prices.index(breakout_price)]

            before_breach = prices[: close_prices.index(breakout_price)]
            post_breach = prices[close_prices.index(breakout_price) :]

            if (len(post_breach) >= self._window2 and
                    all(p > breakout_price for p in post_breach)):
                breakout_symbols.add(symbol)

        weights = {s: 1.0 / len(breakout_symbols) for s in breakout_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
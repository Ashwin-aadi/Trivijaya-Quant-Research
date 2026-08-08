from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "After a breakout occurs, there is often continued upward momentum. This strategy "
        "identifies stocks that have recently broken out and are likely to continue in the same direction."
    )

    def __init__(self, window: int = 20, min_profitability: float = 0.1) -> None:
        self._window = window
        self._min_profitability = min_profitability

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = set()
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            open_prices = [float(v) for v in history[symbol].select("open").drop_nulls().to_list()]
            close_prices = [float(v) for v in history[symbol].select("close").drop_nulls().to_list()]

            if len(open_prices) < self._window or len(close_prices) < self._window:
                continue

            breakout_price = max(close_prices)
            breakout_idx = close_prices.index(breakout_price)

            for i in range(max(0, breakout_idx - 1), min(len(open_prices) - 1, breakout_idx + 2)):
                if (close_prices[i] > open_prices[i]) and (
                    (breakout_price / open_prices[breakout_idx] - 1.0) >= self._min_profitability
                ):
                    breakout_symbols.add(symbol)
                    break

        weights = {s: 1.0 for s in breakout_symbols}
        if not weights:
            return Signal(information_available_at=stamp, weights={})
        else:
            weight_per_symbol = 1.0 / len(breakout_symbols)
            return Signal(
                information_available_at=stamp, weights={s: weight_per_symbol for s in breakout_symbols}
            )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select("session_date").max().unwrap()
    assert isinstance(newest, pl.Date)
    return newest
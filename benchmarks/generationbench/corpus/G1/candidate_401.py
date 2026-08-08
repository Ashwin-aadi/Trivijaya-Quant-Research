from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in the direction of the initial move. "
        "This strategy identifies stocks that have recently broken out and are still trending in the same direction."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            price_series = history[symbol]
            latest_close = float(price_series[-1])
            previous_close = float(price_series[-2])

            # Check if there was a breakout
            if (latest_close > previous_close and previous_close < price_series.rolling_max(window=self._window)) or \
               (latest_close < previous_close and previous_close > price_series.rolling_min(window=self._window)):
                direction = 1 if latest_close > previous_close else -1

                # Check continuation
                for i in range(2, self._window + 1):
                    current_price = float(price_series[-i])
                    if (direction == 1 and current_price < previous_close) or \
                       (direction == -1 and current_price > previous_close):
                        break
                else:
                    breakout_symbols.append(symbol)

        weight = 1.0 / len(breakout_symbols)
        return Signal(information_available_at=stamp, weights={s: weight for s in breakout_symbols})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
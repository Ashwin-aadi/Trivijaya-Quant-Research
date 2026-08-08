from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakout continuation strategies look for a stock that has recently broken out of its "
        "range and then continues to move in the direction of the breakout. This often indicates "
        "that strong momentum is still present."
    )

    def __init__(self, window: int = 20, future_lookback: int = 10) -> None:
        self._window = window
        self._future_lookback = future_lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._future_lookback)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(values) < self._window + 1:
                continue

            high = max(values[-self._window : -1])
            low = min(values[-self._window : -1])
            breakout_price = values[-1]

            if (breakout_price > high and values[-2] <= high) or (
                breakout_price < low and values[-2] >= low
            ):
                continue

            future_prices = [float(v) for v in history[symbol][-self._future_lookback :].to_list()]
            up_trend = all(future_price > break_price for future_price, break_price in zip(future_prices[1:], future_prices[:-1]))
            down_trend = all(future_price < break_price for future_price, break_price in zip(future_prices[1:], future_prices[:-1]))

            if (breakout_price > high and up_trend) or (breakout_price < low and down_trend):
                breakout_symbols.append(symbol)

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
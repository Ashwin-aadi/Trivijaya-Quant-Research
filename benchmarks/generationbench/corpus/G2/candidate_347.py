from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: short-term momentum and "
        "long-term support/resistance levels. If a stock has recently broken through its 50-day "
        "moving average (short-term momentum) while also being near its 200-day moving average, "
        "it suggests strong buying interest could push the stock higher."
    )

    def __init__(self, short_window: int = 50, long_window: int = 200) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._short_window + self._long_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            short_ma = float(history[symbol]["close"].mean().over(window_size=self._short_window))
            long_ma = float(history[symbol]["close"].mean().over(window_size=self._long_window))

            recent_close = float(view.latest_close()[symbol])
            is_breakout = recent_close > short_ma
            is_near_support = abs(recent_close - long_ma) < 0.1 * long_ma

            if is_breakout and is_near_support:
                signals.append(symbol)

        weights = {s: 1 / len(signals) for s in signals} if signals else {}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
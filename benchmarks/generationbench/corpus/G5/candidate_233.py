from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts that continue to new highs after an initial breakout indicate strong "
        "momentum and potential for further gains. This strategy identifies such breakouts."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < (self._window + 1) * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            history = view.history(symbol=symbol).sort("session_date")
            session_dates = [d.to_pydatetime().date() for d in history["session_date"].to_list()]
            close_prices = history["adj_close"].to_list()
            if len(close_prices) < self._window + 1:
                continue

            breakout_day = max(range(self._window), key=lambda i: close_prices[i])
            if (breakout_day == self._window - 1 or
                    close_prices[breakout_day] >= max(close_prices[:breakout_day])):
                next_day_price = close_prices[breakout_day + 1]
                if next_day_price > max(close_prices[breakout_day:]):
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
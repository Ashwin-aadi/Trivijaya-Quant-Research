from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After identifying a breakout candidate, if the stock continues to move in the direction "
        "of the breakout over several sessions, it is more likely to continue this trend. This strategy "
        "identifies such continuation patterns for better entry points."
    )

    def __init__(self, window: int = 20, follow_window: int = 5) -> None:
        self._window = window
        self._follow_window = follow_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window + self._follow_window)
        if closes.height < self._window + self._follow_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window + self._follow_window:
                continue

            breakout_price = max(values[:self._window])
            follow_price = max(values[self._window:self._window + self._follow_window])

            if values[-1] == follow_price and values[-1] > breakout_price:
                breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[:5]
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
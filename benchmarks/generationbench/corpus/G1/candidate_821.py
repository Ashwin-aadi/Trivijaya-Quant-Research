from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a strong breakout, the stock often continues in its direction. This strategy "
        "identifies stocks that have recently broken out and are likely to continue trending."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback)
        if history.height < self._window + self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = set()
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_series = [float(v) for v in history[symbol]["adj_close"].drop_nulls().to_list()]
            high_series = [float(v) for v in history[symbol]["high"].drop_nulls().to_list()]
            low_series = [float(v) for v in history[symbol]["low"].drop_nulls().to_list()]

            if len(close_series) < self._window + self._lookback:
                continue

            breakout_price = max(high_series[-self._lookback:])
            last_close = close_series[-1]
            if last_close > breakout_price:
                breakout_symbols.add(symbol)

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
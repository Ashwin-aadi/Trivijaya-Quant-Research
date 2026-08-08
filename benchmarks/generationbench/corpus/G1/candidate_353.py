from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "After a breakout occurs, the continuation pattern suggests that the price may continue "
        "in the direction of the breakout. This strategy captures potential gains from such a "
        "continuation by identifying symbols that have recently broken out and are still in an "
        "uptrend."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window + self._lookback)
        if closes.height < self._window + self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = set()
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window + self._lookback:
                continue
            breakout_price = max(values[-self._window:])
            continuation_price = max(values[-self._lookback:])
            if values[-1] >= continuation_price and values[-2] < continuation_price:
                if values[-1] > breakout_price:
                    breakout_symbols.add(symbol)

        weights = {s: 1.0 / len(breakout_symbols) for s in breakout_symbols}
        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
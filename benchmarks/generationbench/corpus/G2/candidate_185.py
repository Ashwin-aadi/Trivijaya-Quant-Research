from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a strong upward breakout, there is often a period of consolidation followed by "
        "a continuation of the trend. By identifying symbols that have recently broken out and "
        "are consolidating, we can potentially capture this continuation phase for returns."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback)

        if history.height < self._window + self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].to_list()]
            if len(adj_closes) < self._window + self._lookback:
                continue

            breakout_price = max(adj_closes[-self._window :])
            breakout_date = history["session_date"][adj_closes.index(breakout_price)]

            if adj_closes[adj_closes.index(breakout_price) - 1] < breakout_price and any(
                close > breakout_price for close in adj_closes[
                    adj_closes.index(breakout_price) : adj_closes.index(breakout_price)
                    + self._lookback
                ]
            ):
                consolidation = all(
                    adj_closes[i] <= breakout_price
                    for i in range(adj_closes.index(breakout_price), len(adj_closes))
                    if (adj_closes.index(breakout_price) + 1 < len(adj_closes)) and (
                        adj_closes.index(breakout_price) + self._lookback >= len(adj_closes)
                    )
                )

                if consolidation:
                    breakout_symbols.append(symbol)

        weights = {symbol: 1.0 / len(breakout_symbols) for symbol in breakout_symbols}
        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a strong breakout, stocks often continue in the direction of the breakout. "
        "This strategy identifies such continuation patterns and provides trading signals based on"
        " the relative performance of the stock compared to its recent history."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback)
        if history.height < self._window + self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            hist = history.filter(pl.col("symbol") == symbol).sort(by="session_date")
            recent_closes = [float(v) for v in hist["adj_close"].to_list()[-self._window:]]
            
            if len(recent_closes) < self._window:
                continue

            breakout = max(recent_closes)
            breakout_index = recent_closes.index(breakout)

            if (
                recent_closes[breakout_index - self._lookback] < breakout
                and all(
                    recent_closes[i] > recent_closes[breakout_index - self._lookback]
                    for i in range(breakout_index + 1, len(recent_closes))
                )
            ):
                breakout_symbols.append(symbol)

        weights = {symbol: 1.0 / len(breakout_symbols) for symbol in breakout_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
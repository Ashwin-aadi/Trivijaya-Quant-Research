from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts are often followed by continuation moves. If a stock breaks out to the upside"
        " or downside and then continues that direction over a certain period, it suggests "
        "momentum is still in play, potentially offering opportunities for profits."
    )

    def __init__(self, breakout_window: int = 20, continuation_window: int = 10) -> None:
        self._breakout_window = breakout_window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._breakout_window + self._continuation_window)
        if history.height < self._breakout_window + self._continuation_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            price_series = [float(v) for v in history[symbol].to_list()]
            if len(price_series) < self._breakout_window + self._continuation_window:
                continue

            breakout_price = max(price_series[-self._breakout_window:])
            if any(price >= (1.05 * breakout_price) for price in price_series[-self._continuation_window:]):
                breakout_symbols.append(symbol)

        continuation_symbols: list[str] = []
        for symbol in breakout_symbols:
            history_symbol = view.history(lookback=self._breakout_window + self._continuation_window)[symbol]
            if any(price < (0.95 * max(history_symbol.to_list()[-self._breakout_window:-self._continuation_window])) 
                   for price in history_symbol.to_list()[-self._continuation_window:]):
                continue
            continuation_symbols.append(symbol)

        if not continuation_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(continuation_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in continuation_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
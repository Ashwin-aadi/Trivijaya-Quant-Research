from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price reversion against a trailing reference implies that prices will revert to the "
        "mean after deviating significantly. If recent price levels are far from a moving average, "
        "the market may reverse direction."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        recent_closes = {symbol: float(row[-1]) for symbol, row in zip(view.symbols, history.to_dicts())}
        mean_price = sum(recent_closes.values()) / len(recent_closes)
        
        symbols_with_deviation = []
        for symbol in view.symbols:
            if symbol not in recent_closes:
                continue
            current_close = recent_closes[symbol]
            deviation = abs(current_close - mean_price) / mean_price
            if deviation > 0.1:  # Consider a threshold of 10% as significant deviation
                symbols_with_deviation.append(symbol)

        weight_per_symbol = 1.0 / len(symbols_with_deviation)
        weights = {symbol: weight_per_symbol for symbol in symbols_with_deviation}

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
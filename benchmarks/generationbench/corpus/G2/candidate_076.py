from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class PriceReversion(Strategy):
    rationale = (
        "Price reversion occurs when prices move to extremes and then tend to revert "
        "towards the mean. By identifying symbols that have moved significantly away from "
        "their historical mean price levels, we can generate buy or sell signals based on "
        "the expectation of a return to the mean."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_stats: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_history = [float(v) for v in history[symbol].to_list()]
            mean_price = sum(close_history) / len(close_history)
            latest_close = float(view.latest_close()[symbol])
            deviation = abs(latest_close - mean_price)
            symbol_stats[symbol] = deviation

        sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1], reverse=True)
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_symbol = sorted_symbols[0][0]
        weight = 1.0
        return Signal(
            information_available_at=stamp,
            weights={top_symbol: weight},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
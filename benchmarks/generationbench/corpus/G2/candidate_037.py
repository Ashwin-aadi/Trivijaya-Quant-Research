from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion suggests that stock prices and investment returns will eventually "
        "reverse after a significant deviation from the mean. In short-term trading, "
        "extreme price movements often revert to historical norms."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_prices = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].to_list()]
            mean_price = sum(prices) / self._window
            deviation = abs(prices[-1] - mean_price)
            symbol_prices[symbol] = deviation

        sorted_symbols = [
            s for s, _ in sorted(symbol_prices.items(), key=lambda item: item[1])
        ]
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_symbol = sorted_symbols[0]
        weight = 1.0
        return Signal(
            information_available_at=stamp, weights={top_symbol: weight}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
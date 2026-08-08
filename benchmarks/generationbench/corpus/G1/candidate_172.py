from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reverts to the mean over time. By identifying assets that have significantly "
        "underperformed their historical averages, we can generate buying opportunities."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        mean_price = sum(closes) / len(closes)

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            symbol_history = history.select(["session_date", pl.col(symbol).alias("price")])
            if not symbol_history.height > self._window:
                continue

            prices = [float(v) for v in symbol_history["price"].to_list()]
            mean_price_symbol = sum(prices) / len(prices)
            deviation = abs(mean_price - mean_price_symbol)

            if deviation > 0.15 * mean_price:
                signals[symbol] = 1.0

        return Signal(
            information_available_at=stamp, weights={s: weight for s in signals.keys()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
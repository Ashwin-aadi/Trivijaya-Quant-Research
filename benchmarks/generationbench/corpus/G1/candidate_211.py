from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "High dispersion in the daily range of stock prices can indicate that the market is "
        "volatile and potentially due for a mean reversion. By identifying stocks with high "
        "dispersion, we aim to capitalize on this volatility."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        ranges = []
        for symbol in symbols:
            open_prices = [float(o) for o in history[symbol]["open"].to_list()]
            close_prices = [float(c) for c in history[symbol]["close"].to_list()]
            high_prices = [float(h) for h in history[symbol]["high"].to_list()]
            low_prices = [float(l) for l in history[symbol]["low"].to_list()]

            daily_ranges = [
                (high - low)
                for open_price, close_price, high, low in zip(
                    open_prices, close_prices, high_prices, low_prices
                )
            ]

            mean_range = sum(daily_ranges) / len(daily_ranges)
            dispersion = max(daily_ranges) - min(daily_ranges)

            if dispersion > 2 * mean_range:
                ranges.append(symbol)

        weight = 1.0 / len(ranges)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranges},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
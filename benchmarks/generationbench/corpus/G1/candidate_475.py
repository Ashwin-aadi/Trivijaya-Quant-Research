from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates a market is consolidating and may soon break out. "
        "By identifying symbols with reduced price range over the last 20 days compared to their "
        "previous period, we can identify potential breakout candidates."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        compressed_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            close_prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            recent_range = max(close_prices) - min(close_prices)
            previous_range = (
                max([close_prices[i] for i in range(len(close_prices) - self._window, len(close_prices))])
                - min([close_prices[i] for i in range(len(close_prices) - self._window, len(close_prices))])
            )

            if recent_range < 0.8 * previous_range:
                compressed_symbols.append(symbol)

        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in compressed_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
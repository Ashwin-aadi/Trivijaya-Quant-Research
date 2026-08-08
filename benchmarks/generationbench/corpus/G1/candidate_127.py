from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the market is consolidating within a narrow range. "
        "This can be an indication of potential breakout in either direction, making it a good candidate for entry."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_ranges = {}
        for symbol in view.symbols:
            symbol_data = history.filter(pl.col("symbol") == symbol)
            low_range = (symbol_data["low"].min() - symbol_data["high"].max()).abs()
            close_range = (symbol_data["close"].max() - symbol_data["close"].min()).abs()
            if low_range / close_range < 0.2:
                symbol_ranges[symbol] = low_range / close_range

        selected_symbols = sorted(symbol_ranges, key=lambda x: symbol_ranges[x], reverse=True)[:5]
        weights = {s: 1.0 / len(selected_symbols) for s in selected_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
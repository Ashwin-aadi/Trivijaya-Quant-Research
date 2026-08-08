from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the price is consolidating within a narrow range, "
        "suggesting potential breakout in either direction. This can be an opportune time to enter positions."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or not all(symbol in closes.columns for symbol in view.symbols):
            return Signal(information_available_at=stamp, weights={})

        ranges = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            high_low_range = max(values) - min(values)
            ranges[symbol] = high_low_range

        if not ranges:
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = sorted(ranges.keys(), key=lambda x: ranges[x], reverse=True)[:5]
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
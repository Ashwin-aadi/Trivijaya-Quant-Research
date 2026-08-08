from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression signals strong momentum and can indicate a potential reversal or continuation "
        "of the current trend. By identifying stocks where recent price action has been more subdued compared to "
        "historical ranges, we can potentially identify opportunities for entry or exit."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_range = {}
        for symbol in view.symbols:
            if symbol not in history.columns or len(history[symbol].unique()) < 2:
                continue
            high_min = float(history[symbol].max())
            low_max = float(history[symbol].min())
            range_ratio = (high_min - low_max) / (high_min + low_max)
            symbol_range[symbol] = range_ratio

        sorted_ranges = sorted(symbol_range.items(), key=lambda x: x[1])
        top_symbols = [s for s, r in sorted_ranges[:5]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
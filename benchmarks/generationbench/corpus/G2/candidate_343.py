from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompressionStrategy(Strategy):
    rationale = (
        "Range compression indicates that a stock's price is oscillating within a tighter range "
        "than its historical volatility. This can be a sign of underlying strength or weakness, "
        "and may suggest an opportunity to enter positions before the market breaks out in one direction."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        range_compression_scores = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            high_values = [float(v) for v in history[symbol + "_high"].drop_nulls().to_list()]
            low_values = [float(v) for v in history[symbol + "_low"].drop_nulls().to_list()]

            if len(high_values) < self._window or len(low_values) < self._window:
                continue

            max_high = max(high_values)
            min_low = min(low_values)

            high_range = max_high - min_low
            mean_close = sum(float(v) for v in history[symbol + "_close"].to_list()) / len(
                history
            )

            compression_ratio = (high_range / mean_close) * 100

            range_compression_scores[symbol] = compression_ratio

        sorted_symbols = [
            symbol for _, symbol in sorted(range_compression_scores.items(), key=lambda item: item[1])
        ]

        if not range_compression_scores:
            return Signal(information_available_at=stamp, weights={})

        top_n_symbols = sorted_symbols[:5]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that prices are moving in a tighter range relative to their "
        "historical volatility. This can be an indication of a potential breakout or reversal, "
        "as tight ranges often precede significant price movements."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        range_compression_scores = []
        for symbol in symbols:
            high_min = history.select(pl.col(symbol).max()).item()
            low_max = -history.select(pl.col(symbol).min()).item()  # Convert to positive
            total_range = high_min + low_max
            mean_price = (high_min + low_max) / 2
            range_compression_score = (mean_price - min(high_min, low_max)) / total_range

            range_compression_scores.append((symbol, range_compression_score))

        # Sort by range compression score in descending order to identify symbols with the most compressed ranges
        sorted_symbols = sorted(range_compression_scores, key=lambda x: x[1], reverse=True)

        top_n_symbols = [symbol for symbol, _ in sorted_symbols[:5]]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).item()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates increased investor sentiment and reduced volatility. "
        "Such periods often precede price breakout or consolidation. By identifying stocks with high range "
        "compression, we can benefit from potential subsequent moves."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty() or history.height < (2 * self._window + 1):
            return Signal(information_available_at=stamp, weights={})

        compression_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            high_series = history[symbol].select("high").to_list()
            low_series = history[symbol].select("low").to_list()

            if len(high_series) < self._window or len(low_series) < self._window:
                continue

            highest_high = max(high_series)
            lowest_low = min(low_series)

            range_compression = (highest_high - lowest_low) / highest_high
            compression_scores[symbol] = range_compression

        top_symbols = sorted(compression_scores, key=compression_scores.get, reverse=True)[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
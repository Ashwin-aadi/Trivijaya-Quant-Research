from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompressionStrategy(Strategy):
    rationale = (
        "Range compression occurs when price volatility decreases, indicating that the market "
        "is consolidating. Such periods often precede breakout moves or consolidation patterns, "
        "potentially offering profitable entry points into a range-bound market."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_data = {}
        for symbol in view.symbols:
            hist = history.filter(pl.col("symbol") == symbol).sort("session_date")
            high_close_ratio = (hist["high"].max() - hist["close"]) / hist["close"].min()
            low_close_ratio = (hist["low"].min() - hist["close"]) / hist["close"].max()
            compression_ratio = min(high_close_ratio, low_close_ratio)
            symbol_data[symbol] = {"compression": compression_ratio}

        compressed_symbols = [s for s, d in symbol_data.items() if d["compression"] < self._threshold]
        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in compressed_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
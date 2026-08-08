from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests a market is consolidating and may be due for a breakout. "
        "We target stocks where the recent high-low range has significantly shrunk."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        compressed_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns or "session_date" not in history.columns:
                continue

            highs = [float(v) for v in history.select(pl.col(symbol).max()).to_series()]
            lows = [float(v) for v in history.select(pl.col(symbol).min()).to_series()]
            ranges = [(high - low) / 2.0 for high, low in zip(highs, lows)]
            latest_range = ranges[-1]

            if len(ranges) < self._window:
                continue

            avg_range = sum(ranges) / len(ranges)
            compression_ratio = latest_range / avg_range
            if compression_ratio <= 0.5:
                compressed_symbols.append(symbol)

        weights: dict[str, float] = {s: 1.0 for s in compressed_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest
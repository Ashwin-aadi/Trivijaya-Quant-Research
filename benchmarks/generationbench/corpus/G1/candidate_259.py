from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates increased trading activity and reduced volatility, "
        "suggesting that the market is preparing for a breakout or a significant move in either direction."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        compressed_ranges: list[tuple[str, float]] = []
        for symbol in view.symbols:
            opens = [float(v) for v in history[symbol + "_open"].to_list()]
            closes = [float(v) for v in history[symbol + "_close"].to_list()]
            if len(opens) < self._window or len(closes) < self._window:
                continue
            highs = [float(v) for v in history[symbol + "_high"].to_list()]
            lows = [float(v) for v in history[symbol + "_low"].to_list()]
            range_compression = (max(highs[-self._window:]) - min(lows[-self._window:])) / (
                max(closes[-self._window:]) - min(opens[-self._window:])
            )
            compressed_ranges.append((symbol, range_compression))

        if not compressed_ranges:
            return Signal(information_available_at=stamp, weights={})

        sorted_ranges = sorted(compressed_ranges, key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in sorted_ranges[:5]]
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
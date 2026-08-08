from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests a market is consolidating and may be due for a breakout. "
        "We target stocks where the recent price range has significantly narrowed."
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
            if symbol not in history.columns:
                continue
            data = history[[symbol]].sort("session_date")
            highs = [float(h) for h in data["high"].to_list()]
            lows = [float(l) for l in data["low"].to_list()]
            range_ratio = (max(highs) - min(lows)) / max(highs)
            if range_ratio < 0.1:
                compressed_symbols.append(symbol)

        weights = {s: 1.0 / len(compressed_symbols) for s in compressed_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
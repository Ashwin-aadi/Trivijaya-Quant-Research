from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "During periods of range compression, the price movement is more confined. "
        "This can indicate a potential trend reversal or consolidation phase, making it "
        "opportune to trade based on the relative strength within the compressed range."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        range_compression_ratio = (
            (history["high"] - history["low"]) / abs(history["close"].mean())
        ).to_list()
        mean_ratio = sum(range_compression_ratio) / len(range_compression_ratio)
        
        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        compressed_symbols: list[str] = []
        for symbol in symbols:
            ratio = range_compression_ratio[symbols.index(symbol)]
            if ratio < mean_ratio * 0.75:
                compressed_symbols.append(symbol)

        weights = {s: 1.0 / len(compressed_symbols) for s in compressed_symbols}
        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
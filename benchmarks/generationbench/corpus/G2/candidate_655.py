from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the price action has been more volatile within a "
        "narrower range than usual. This could suggest an upcoming breakout or consolidation, "
        "potentially leading to larger future price movements."
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
            data = history.select(["session_date", f"{symbol}"]).to_pandas()
            highs = data[symbol].max()
            lows = data[symbol].min()
            range_ratio = (highs - lows) / highs
            if range_ratio < self._threshold:
                symbol_data[symbol] = range_ratio

        if not symbol_data:
            return Signal(information_available_at=stamp, weights={})

        picks = sorted(symbol_data.keys(), key=lambda s: symbol_data[s], reverse=True)
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
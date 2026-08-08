from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when a stock's price moves within a narrow range for an "
        "extended period. This suggests that the market is uncertain about the future direction "
        "of the asset and could potentially reverse soon. By identifying stocks with high range "
        "compression, we can capture potential reversals."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        range_compression_ratio = (
            (history["high"] - history["low"]) / history["close"].shift(-1)
        ).mean().item()
        if range_compression_ratio < 0.25:  # Consider this a threshold
            symbols_with_high_range_compression = [
                symbol for symbol in view.symbols
                if (view.closes(lookback=self._window)[symbol] / history["close"].shift(-1)).mean().item() < 0.25
            ]
            if not symbols_with_high_range_compression:
                return Signal(information_available_at=stamp, weights={})
            weight = 1.0 / len(symbols_with_high_range_compression)
            return Signal(
                information_available_at=stamp,
                weights={s: weight for s in symbols_with_high_range_compression},
            )
        else:
            return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
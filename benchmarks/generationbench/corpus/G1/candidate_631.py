from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression20d(Strategy):
    rationale = (
        "Range compression indicates that the market is consolidating or entering a trading range. "
        "This can be a signal for potential breakout in either direction and may present opportunities for trade."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        opens = [float(v) for v in history["open"].to_list()]
        closes = [float(v) for v in history["close"].to_list()]

        def range_compression_ratio(open_val: float, close_val: float) -> float:
            return (abs(close_val - open_val)) / max(abs(open_val), abs(close_val))

        ratios = [
            range_compression_ratio(opens[i], closes[i])
            for i in range(len(symbols))
        ]
        
        picks: list[str] = [symbols[i] for i, r in enumerate(ratios) if r >= self._threshold]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
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
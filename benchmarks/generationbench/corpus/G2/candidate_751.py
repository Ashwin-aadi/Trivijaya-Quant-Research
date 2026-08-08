from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when the trading range of a stock narrows significantly. "
        "This suggests that market participants are becoming less certain about the future price direction, "
        "potentially leading to mean reversion or breakout in either direction. "
        "Stocks with high range compression could be good candidates for investment."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            low_high_ratio = (history.select(pl.col("high") - pl.col("low")).max() /
                              history.select(pl.col("close").shift(-1) - pl.col("open")).max()).item()
            if low_high_ratio < 0.5:
                signals[symbol] = 1.0 / len(signals)

        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
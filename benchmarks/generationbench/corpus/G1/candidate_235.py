from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the stock price has been consolidating and "
        "may soon break out. By identifying symbols with reduced volatility recently, we can "
        "potentially benefit from a breakout in either direction."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = [float(v) for v in view.closes().drop_nulls().to_list()]
        if len(closes) < self._window:
            return Signal(information_available_at=stamp, weights={})

        avg_close = sum(closes[-self._window:]) / self._window
        range_compression = [
            abs(float(high) - float(low)) / (float(close) + 1e-8)
            for symbol, high, low, close in zip(
                history["symbol"],
                history["high"].to_list(),
                history["low"].to_list(),
                history["close"].to_list(),
            )
        ]
        compressed = sorted(zip(range_compression, [s for s in view.symbols]), reverse=True)

        if not compressed or len(compressed) < 5:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [symbol for _, symbol in compressed[:5]]
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
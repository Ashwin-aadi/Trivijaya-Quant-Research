from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the market is consolidating and may be ready for "
        "a breakout in either direction. By identifying symbols with reduced price range over a "
        "recent period, we can prepare to enter trades when the market breaks out."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            hist = history.filter(pl.col("symbol") == symbol).sort("session_date")
            open_prices = [float(v) for v in hist["open"].to_list()]
            close_prices = [float(v) for v in hist["close"].to_list()]

            if len(open_prices) < self._window or len(close_prices) < self._window:
                continue

            high = max([float(v) for v in hist["high"].to_list()])
            low = min([float(v) for v in hist["low"].to_list()])
            range_ratio = (high - low) / close_prices[-1]

            if range_ratio <= self._threshold:
                picks.append(symbol)

        picks = picks[:5]
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
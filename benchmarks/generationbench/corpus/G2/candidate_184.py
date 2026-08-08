from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price reversion strategies seek to profit from the tendency of prices to return to a "
        "previous level after moving away. In this case, we use a trailing reference price (e.g.,"
        "50-day simple moving average) and buy stocks that have fallen significantly below their"
        "trailing mean."
    )

    def __init__(self, window: int = 50, threshold: float = 1.2) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        symbols = closes.columns[1:]  # Skip session_date column
        trailing_means: dict[str, float] = {}
        for symbol in symbols:
            values = [float(v) for v in history[symbol].to_list()]
            mean = sum(values) / len(values)
            trailing_means[symbol] = mean

        signals: list[tuple[str, float]] = []
        for symbol in symbols:
            latest_close = view.latest_close()[symbol]
            if latest_close < trailing_means[symbol] * (1.0 - self._threshold):
                weight = 1.0 / len(symbols)
                signals.append((symbol, weight))

        if not signals:
            return Signal(information_available_at=stamp, weights={})
        return Signal(
            information_available_at=stamp,
            weights=dict(signals),
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
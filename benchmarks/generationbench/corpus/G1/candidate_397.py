from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion occurs when an asset's price returns to the historical average. "
        "In a short horizon, extreme deviations from this mean can provide profitable entry points."
    )

    def __init__(self, window: int = 10, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = [float(v) for v in history["adj_close"].to_list()]
        mean = sum(closes) / len(closes)
        deviations = [(c - mean) for c in closes]

        if not all(abs(d) < self._threshold * abs(mean) for d in deviations):
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
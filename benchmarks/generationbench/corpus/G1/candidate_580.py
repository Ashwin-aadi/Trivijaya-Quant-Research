from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reverts to the mean over time. By identifying stocks that have deviated significantly "
        "from their 20-day average, we can identify potential reversals and profit from this tendency."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_closes = closes.mean()
        deviations = {s: abs(c - m) for s, c, m in zip(
            mean_closes.columns,
            [float(mean_closes[s].item()) for s in mean_closes.columns],
            [float(mean_closes[s].mean().item()) for s in mean_closes.columns]
        )}

        revertibles = [
            symbol for symbol, deviation in deviations.items()
            if deviation > self._threshold * float(deviation)
        ]

        if not revertibles:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(revertibles)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in revertibles}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
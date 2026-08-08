from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends based on volatility. High recent volatility suggests "
        "that the market is uncertain and potentially more prone to trend continuation or reversal. "
        "We enter into a position if the current close is above (or below) its mean over a period, "
        "indicating a strong trend."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = [float(v) for v in view.closes().drop_nulls().to_list()]
        mean_close = sum(closes[-self._window:]) / self._window
        std_dev_close = (sum((c - mean_close) ** 2 for c in closes[-self._window:]) /
                         self._window) ** 0.5

        if abs(view.latest_close()[view.symbols[0]] - mean_close) >= std_dev_close * self._threshold:
            direction = "long" if view.latest_close()[view.symbols[0]] > mean_close else "short"
            return Signal(
                information_available_at=stamp,
                weights={s: 1.0 for s in view.symbols},
                market_view=view.market_view(),
            )

        return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
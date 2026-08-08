from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Reversion to the mean suggests that prices which have deviated significantly from "
        "their historical average will tend to revert back. This strategy identifies stocks "
        "that are currently trading far below their trailing 20-day average and allocates "
        "capital to these undervalued assets."
    )

    def __init__(self, window: int = 20, threshold: float = 0.8) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = [float(v) for v in history["adj_close"].to_list()]
        mean_close = sum(closes[:-1]) / (self._window)
        latest_close = float(history["adj_close"][history.height - 1])

        if latest_close / mean_close < self._threshold:
            return Signal(
                information_available_at=stamp,
                weights={s: 0.5 for s in view.symbols},
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
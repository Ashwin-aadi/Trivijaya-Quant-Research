from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality in financial markets often reflects predictable patterns driven by "
        "calendar effects. For instance, certain stocks may perform well during specific months "
        "of the year due to holidays, government policies, or cultural events."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or len(history["symbol"].unique()) < 5:
            return Signal(information_available_at=stamp, weights={})

        symbol = view.symbols[0]  # Assuming all symbols have the same seasonal pattern
        df = history.filter(pl.col("session_date").dt.month() == stamp.month)

        if df.height < self._window / 12:  # Check if enough data points are available for this month
            return Signal(information_available_at=stamp, weights={})

        recent_closes = df.select("adj_close")
        mean_close = float(recent_closes.mean().item())

        if view.latest_close()[symbol] > mean_close:
            weight = 1.0 / len(view.symbols)
            return Signal(
                information_available_at=stamp,
                weights={s: weight for s in view.symbols},
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
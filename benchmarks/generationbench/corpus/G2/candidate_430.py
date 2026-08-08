from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Certain times of the year may exhibit higher trading activity or specific market "
        "patterns due to seasonal effects such as fiscal year-end periods, holidays, and "
        "weather. By identifying these trends, we can take advantage of historical price "
        "behavior during these periods."
    )

    def __init__(self, window: int = 30, seasonality_window: int = 90) -> None:
        self._window = window
        self._seasonality_window = seasonality_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._seasonality_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        seasonality = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            daily_returns = (
                (history[symbol].to_series() / history[symbol].shift(1).to_series() - 1.0)
                .drop_nulls()
                .to_list()
            )
            seasonality[symbol] = max(daily_returns[-self._window :])

        top_symbols = sorted(seasonality.items(), key=lambda x: x[1], reverse=True)[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [symbol for symbol, _ in top_symbols]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
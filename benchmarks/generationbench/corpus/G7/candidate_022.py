from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Annual seasonal effects can be exploited by identifying stocks that exhibit "
        "significant price movements at certain times of the year. By holding a diversified "
        "portfolio during these periods, we aim to capture positive returns."
    )

    def __init__(self, lookback_years: int = 5, window: int = 365, top_n: int = 10) -> None:
        self._lookback_years = lookback_years
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_years * 365)
        if closes.height < self._lookback_years * 365:
            return Signal(information_available_at=stamp, weights={})

        seasonal_changes: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._lookback_years * 365:
                continue
            mean_close = sum(values[-self._window:]) / self._window
            annual_change = (values[-1] - values[0]) / values[0]
            seasonal_changes[symbol] = annual_change

        sorted_changes = sorted(seasonal_changes.items(), key=lambda x: x[1], reverse=True)
        picks = [s for s, _ in sorted_changes[: self._top_n]]
        
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
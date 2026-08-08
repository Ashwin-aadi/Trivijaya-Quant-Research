from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Certain equities exhibit stronger performance during specific times of the year. "
        "By identifying these seasonal patterns, we can allocate capital more effectively."
    )

    def __init__(self, window: int = 365, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_effect: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            daily_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            yearly_closes = [daily_closes[i - i % 365] for i in range(len(daily_closes))]
            mean_close = sum(yearly_closes) / len(yearly_closes)
            current_year_close = daily_closes[-1]
            if abs(current_year_close - mean_close) / mean_close > self._threshold:
                seasonal_effect[symbol] = 1.0

        if not seasonal_effect:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(seasonal_effect)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in seasonal_effect},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
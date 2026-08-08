from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "Certain stocks exhibit strong performance during specific seasons or months of the year. "
        "By identifying and leveraging these seasonal effects, we can potentially achieve better returns."
    )

    def __init__(self, window: int = 60, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            monthly_closes = _group_by_month(values)
            seasonal_effect = max(monthly_closes.values()) / min(monthly_closes.values()) - 1.0
            if seasonal_effect >= 0.2:  # Define a threshold for strong seasonal effect
                picks.append(symbol)

        picks = picks[: self._top_n]
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


def _group_by_month(values: list[float]) -> dict[int, float]:
    monthly_closes = {}
    for i in range(len(values)):
        month = (view.as_of.year - values[i] // 100) * 12 + (view.as_of.month - 1)
        if month not in monthly_closes:
            monthly_closes[month] = []
        monthly_closes[month].append(values[i])
    for month, closes in monthly_closes.items():
        monthly_closes[month] = max(closes) / min(closes) - 1.0
    return monthly_closes
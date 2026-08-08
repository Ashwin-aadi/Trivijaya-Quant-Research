from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Seasonal effects in equity markets can arise due to various factors such as "
        "weather patterns, holiday calendars, or specific economic activities. This strategy "
        "exploits historical trends by buying stocks that perform well during certain months."
    )

    def __init__(self, window: int = 20, seasonal_months: tuple[int, ...] = (12,)) -> None:
        self._window = window
        self._seasonal_months = seasonal_months

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        seasonality_factors: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate seasonality factor based on recent performance during seasonal months
            month_factors = [
                1.0 if m == view.as_of.month else 0.0 for m in range(1, 13)
            ]
            seasonality_factor = max(
                (values[i] - min(values)) / (max(values) - min(values))
                * month_factors[i]
                for i in range(self._window)
            )
            seasonality_factors[symbol] = seasonality_factor

        top_symbols = sorted(seasonality_factors, key=seasonality_factors.get, reverse=True)[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
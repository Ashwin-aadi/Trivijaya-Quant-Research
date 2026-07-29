from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Equities often exhibit seasonality based on calendar effects. Certain periods may "
        "see more buying or selling behavior due to external factors like holidays, earnings "
        "seasons, and macroeconomic events. This strategy seeks to capture these trends by "
        "identifying symbols with historically strong performance in specific months."
    )

    def __init__(self, window: int = 120, threshold: float = 0.05) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            mean_close = sum(values) / len(values)
            recent_close = max(values[-12:], default=0.0)
            trend = (recent_close - mean_close) / mean_close
            if abs(trend) >= self._threshold:
                seasonal_trends[symbol] = trend

        top_symbols = sorted(seasonal_trends, key=seasonal_trends.get, reverse=True)[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
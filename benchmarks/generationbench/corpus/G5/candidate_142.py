from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Certain stocks exhibit stronger performance during specific months or seasons. "
        "By identifying these seasonal trends and focusing on symbols with significant "
        "performance boosts in those months, we can construct a strategy that takes advantage of "
        "the calendar effects in the Indian market."
    )

    def __init__(self, window: int = 90) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        seasonal_trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            monthly_closes = {d.month: c for d, c in zip(history["session_date"].dt.month(), closes)}
            avg_monthly_close = sum(monthly_closes.values()) / len(monthly_closes)
            trend_scores = [c - avg_monthly_close for c in monthly_closes.values()]
            max_trend_score = max(trend_scores) if trend_scores else 0
            seasonal_trends[symbol] = max_trend_score

        sorted_symbols = [
            s for s, _ in sorted(seasonal_trends.items(), key=lambda x: x[1], reverse=True)
        ]
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = sorted_symbols[:3]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().date()
    assert isinstance(newest, date)
    return newest
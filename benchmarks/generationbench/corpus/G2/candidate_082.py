from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "Certain stocks may exhibit higher returns during specific seasons or months of the year. "
        "By identifying and exploiting these seasonal effects, investors can potentially achieve "
        "above-average returns."
    )

    def __init__(self, window: int = 365, lookback_months: int = 12) -> None:
        self._window = window
        self._lookback_months = lookback_months

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or (stamp - date.fromisoformat(history["session_date"].max())).days < self._lookback_months * 30.44:
            return Signal(information_available_at=stamp, weights={})

        seasonal_factors: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            monthly_closes = [closes[i * 30 : (i + 1) * 30] for i in range(len(closes) // 30)]
            avg_monthly_returns = [(m[-1] - m[0]) / m[0] for m in monthly_closes]
            if not avg_monthly_returns:
                continue
            seasonal_factors[symbol] = max(avg_monthly_returns)

        top_symbols = sorted(seasonal_factors.items(), key=lambda x: x[1], reverse=True)[:self._lookback_months]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
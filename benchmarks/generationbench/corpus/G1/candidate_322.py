from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Certain stocks in the Indian market exhibit stronger performance during specific "
        "seasons or months of the year. By identifying and capitalizing on these seasonal trends, "
        "we can generate alpha by timing our entries and exits."
    )

    def __init__(self, window: int = 365, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_trends = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            daily_returns = (
                (history[symbol].to_list()[1:] / history[symbol].to_list()[:-1]) - 1.0
            )
            monthly_returns = [
                sum(daily_returns[i : i + 21])
                for i in range(0, len(daily_returns), 21)
            ]
            avg_monthly_return = sum(monthly_returns) / len(monthly_returns)

            if abs(avg_monthly_return) > self._threshold:
                seasonal_trends[symbol] = avg_monthly_return

        sorted_trends = sorted(seasonal_trends.items(), key=lambda x: abs(x[1]), reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_trends]

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
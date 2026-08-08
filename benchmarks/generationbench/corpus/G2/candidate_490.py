from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality in stock markets refers to patterns that recur at regular intervals "
        "and are related to specific times of the year. For example, certain sectors may "
        "perform better during festive seasons or when weather conditions are favorable. "
        "This strategy aims to capture these periodic returns by identifying symbols that "
        "have historically performed well in a given month."
    )

    def __init__(self, seasonality_window: int = 12) -> None:
        self._seasonality_window = seasonality_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._seasonality_window)
        if closes.height < self._seasonality_window or len(closes.columns) - 1 < 2:
            return Signal(information_available_at=stamp, weights={})

        # Compute the average monthly returns
        avg_monthly_returns: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            monthly_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(monthly_closes) < self._seasonality_window:
                continue

            # Calculate the average return per month
            avg_returns = [
                (monthly_closes[i + 1] - monthly_closes[i]) / monthly_closes[i]
                for i in range(len(monthly_closes) - 1)
            ]
            if len(avg_returns) < 12:
                continue

            avg_monthly_return = sum(avg_returns) / 12
            avg_monthly_returns[symbol] = avg_monthly_return

        # Identify the best performing symbols
        top_symbols = sorted(
            avg_monthly_returns.items(), key=lambda x: x[1], reverse=True
        )[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s, _ in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
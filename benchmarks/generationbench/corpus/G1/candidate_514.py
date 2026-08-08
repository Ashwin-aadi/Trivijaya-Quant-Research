from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Certain months of the year exhibit higher trading volumes and "
        "greater price movements due to seasonal effects. This strategy aims to "
        "capitalize on these anomalies by focusing on sectors or stocks that show "
        "historical strength during specific times of the year."
    )

    def __init__(self, lookback_years: int = 5) -> None:
        self._lookback_years = lookback_years

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_years * 252)  # Assuming 252 trading days in a year
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        seasonal_strength = {}
        for symbol in symbols:
            close_prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            month_groups = _group_by_month(close_prices)
            monthly_strength = {m: sum(prices) / len(prices) for m, prices in month_groups.items()}
            seasonal_strength[symbol] = max(monthly_strength.values())

        top_symbols = sorted(seasonal_strength.keys(), key=lambda k: seasonal_strength[k], reverse=True)[:5]
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


def _group_by_month(prices):
    grouped = {}
    for i in range(len(prices)):
        month = (i // 252) % 12 + 1
        if month not in grouped:
            grouped[month] = []
        grouped[month].append(prices[i])
    return grouped
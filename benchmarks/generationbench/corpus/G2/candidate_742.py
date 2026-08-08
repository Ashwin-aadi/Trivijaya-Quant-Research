from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality in the Indian stock market can arise from various factors such as "
        "fiscal year end effects, government policies, and investor behavior. By identifying "
        "stocks that exhibit strong performance during specific times of the year, we can "
        "capitalize on these trends."
    )

    def __init__(self, lookback_years: int = 5) -> None:
        self._lookback_years = lookback_years

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_years * 365)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        recent_closes = pl.DataFrame(history["session_date"].to_list()[-20:], orientation="row")
        symbol_close_history: dict[str, list[float]] = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            symbol_close_history[symbol] = values[-20:]

        # Calculate the mean close over the lookback period for each stock
        means: dict[str, float] = {symbol: sum(values) / len(values) for symbol, values in symbol_close_history.items() if len(values) >= 20}

        # Identify symbols with above average performance in recent history
        cutoff_mean = sum(means.values()) / len(means)
        strong_performers = [s for s, m in means.items() if m > cutoff_mean]

        if not strong_performers:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(strong_performers)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in strong_performers},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
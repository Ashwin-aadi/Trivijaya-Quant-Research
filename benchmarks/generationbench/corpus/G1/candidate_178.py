from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality in stock markets refers to patterns that occur at specific times of the year. "
        "Some stocks exhibit stronger performance during certain months or quarters due to various factors like "
        "weather conditions, corporate events, or macroeconomic trends. Capturing these seasonal effects can provide "
        "trading opportunities."
    )

    def __init__(self, window: int = 120) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_strengths: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            monthly_closes = history.select(
                pl.col("session_date").dt.month(), pl.col(symbol)
            ).group_by("session_date.dt.month()").agg(pl.col(symbol).mean()).collect()
            monthly_averages = [float(row[symbol]) for row in monthly_closes]
            max_avg = max(monthly_averages)
            if max_avg == 0.0:
                continue
            seasonal_strengths[symbol] = history.filter(
                pl.col("session_date").dt.month() == monthly_closes.select(pl.col("month()")).max()
            )[symbol].mean().to_list()[0] / max_avg

        top_symbols = sorted(seasonal_strengths, key=seasonal_strengths.get, reverse=True)[:5]
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
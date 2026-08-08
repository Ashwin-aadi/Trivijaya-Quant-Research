from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Certain stocks in India exhibit seasonal patterns in their performance. "
        "By identifying these patterns and trading accordingly, we can take advantage of "
        "the predictable swings in stock prices."
    )

    def __init__(self, window: int = 365, lookback_periods: int = 4) -> None:
        self._window = window
        self._lookback_periods = lookback_periods

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history.select(
            pl.col("session_date").dt.year(),
            pl.col("adj_close"),
        ).pivot(index="session_date", columns="year", values="adj_close")

        seasonal_patterns: dict[int, float] = {}
        for year in set(closes.columns) - {"session_date"}:
            avg_closes = closes[["session_date", year]].group_by(
                "session_date"
            ).agg(pl.col(year).mean()).sort("session_date")
            if avg_closes.height >= self._lookback_periods:
                seasonal_patterns[int(year)] = float(avg_closes.sort("session_date").select(
                    pl.col(year).rank(method="dense", descending=True)
                )[0, 0])

        if not seasonal_patterns:
            return Signal(information_available_at=stamp, weights={})

        symbol_weights: dict[str, float] = {}
        for symbol in view.symbols:
            year = history.select(pl.col("session_date").dt.year()).filter(
                pl.col("symbol") == symbol
            ).tail(self._lookback_periods).select(
                pl.col(0).last()
            )[0, 0]
            if year not in seasonal_patterns:
                continue

            rank = seasonal_patterns[year]
            symbol_weights[symbol] = (2.0 / len(view.symbols)) * rank

        return Signal(information_available_at=stamp, weights=symbol_weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
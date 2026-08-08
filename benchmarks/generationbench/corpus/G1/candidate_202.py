from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality effects can provide predictive power in equity markets. By analyzing "
        "the historical performance of stocks around specific dates or seasons, we can "
        "identify periods when certain stocks tend to outperform others."
    )

    def __init__(self, window: int = 30, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        avg_returns: dict[str, float] = {}
        for symbol in view.symbols:
            closes = [float(v) for v in view.closes(symbol).to_list()]
            if len(closes) < self._window:
                continue
            windowed_closes = history.select(
                pl.col("symbol").eq(symbol)
            ).select(pl.col("adj_close"))
            returns = (windowed_closes.to_series() / windowed_closes.shift(1).to_series() - 1.0).to_list()
            avg_return = sum(returns) / len(returns)
            if abs(avg_return) >= self._threshold:
                avg_returns[symbol] = avg_return

        if not avg_returns:
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = sorted(avg_returns.keys(), key=lambda s: avg_returns[s], reverse=True)[:5]
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
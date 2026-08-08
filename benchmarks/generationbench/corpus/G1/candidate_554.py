from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy aims to identify trending stocks by measuring the volatility of price "
        "changes over a rolling window and adjusting weights based on trends. High volatility "
        "in a rising or falling trend indicates strong momentum, which is exploited for gains."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            close_series = history.select(pl.col("adj_close")).to_pandas()[symbol]
            trend_score = self._calculate_trend_score(close_series)
            if abs(trend_score) > self._threshold:
                trends[symbol] = trend_score

        weights: dict[str, float] = {}
        for symbol, score in trends.items():
            weight = 1.0 / len(trends)
            if score > 0:
                weights[symbol] = weight * (1 + abs(score))
            else:
                weights[symbol] = -weight * abs(score)

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_trend_score(close_series: list[float]) -> float:
    returns = [close_series[i] / close_series[i - 1] - 1 for i in range(1, len(close_series))]
    mean_return = sum(returns) / len(returns)
    volatility = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5
    return mean_return / volatility if volatility != 0 else 0
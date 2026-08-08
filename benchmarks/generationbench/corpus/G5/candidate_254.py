from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy aims to follow trends while scaling positions based on volatility. "
        "High volatility periods suggest increased risk and thus lower position sizes, whereas "
        "low volatility periods allow for larger positions."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        signals = []
        for symbol in symbols:
            close_prices = _extract_close_prices(history, symbol)
            trend = _calculate_trend(close_prices)
            volatility = _calculate_volatility(close_prices)

            weight = min(1.0, 2 * (trend / volatility))
            signals.append((symbol, weight))

        return Signal(
            information_available_at=stamp,
            weights=dict(signals),
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _extract_close_prices(history: pl.DataFrame, symbol: str) -> list[float]:
    close_prices = history.select(pl.col("adj_close")[history["symbol"] == symbol]).to_series().to_list()
    if not close_prices:
        return []
    return [float(v) for v in close_prices[0]]


def _calculate_trend(prices: list[float]) -> float:
    returns = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
    positive_returns = sum([r > 0 for r in returns])
    trend = positive_returns / max(len(returns), 1)
    return trend


def _calculate_volatility(prices: list[float]) -> float:
    returns = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
    volatility = (sum([r**2 for r in returns]) / max(len(returns), 1)) ** 0.5
    return volatility
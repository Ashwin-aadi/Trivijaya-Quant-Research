from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy follows the trend of high-volatility stocks. "
        "High volatility suggests increased market activity and potentially higher returns."
    )

    def __init__(self, window: int = 20, threshold_multiplier: float = 1.5) -> None:
        self._window = window
        self._threshold_multiplier = threshold_multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatilities: list[float] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            returns = [(values[i + 1] - values[i]) / values[i] for i in range(len(values) - 1)]
            mean_return = sum(returns) / (len(returns))
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = (variance * self._window) ** 0.5
            volatilities.append(volatility)

        if not volatilities:
            return Signal(information_available_at=stamp, weights={})

        threshold = max(volatilities) * self._threshold_multiplier
        trending_symbols = [symbol for symbol, volatility in zip(view.symbols, volatilities) if volatility >= threshold]

        if not trending_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(trending_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in trending_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
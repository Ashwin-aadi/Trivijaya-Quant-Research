from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends based on volatility. High volatility periods are expected "
        "to continue trending in the same direction, allowing for profitable entries."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or not closes.rows():
            return Signal(information_available_at=stamp, weights={})

        volatility_factors = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            close_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(close_series) < self._window:
                continue

            returns = [(close_series[i] - close_series[i-1]) / close_series[i-1]
                       for i in range(1, len(close_series))]
            mean_return = sum(returns) / len(returns)
            std_dev = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5
            volatility_factor = std_dev / abs(mean_return)

            if volatility_factor > self._threshold:
                volatility_factors.append(symbol)

        volatility_factors = list(set(volatility_factors))[:10]
        if not volatility_factors:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(volatility_factors)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in volatility_factors}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
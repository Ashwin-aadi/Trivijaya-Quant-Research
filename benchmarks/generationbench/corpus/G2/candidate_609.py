from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility ones over the long term. "
        "This is because lower volatility often correlates with steadier earnings growth and "
        "lower risk, which can result in higher returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatilities = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue

            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = (sum([r**2 for r in returns]) / len(returns)) ** 0.5
            volatilities.append((symbol, volatility))

        sorted_volatilities = sorted(volatilities, key=lambda x: x[1])
        low_vol_symbols = [s for s, v in sorted_volatilities[:int(len(sorted_volatilities) * 0.3)]]

        if not low_vol_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(low_vol_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in low_vol_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
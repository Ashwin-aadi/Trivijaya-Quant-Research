from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to capture trends while adjusting position "
        "sizes based on the recent volatility of assets. Higher volatility periods suggest "
        "a cautious approach, whereas lower volatility indicates potential for larger positions."
    )

    def __init__(self, window: int = 20, scale_factor: float = 1.5) -> None:
        self._window = window
        self._scale_factor = scale_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        open_prices = [float(v) for v in history[symbols[0]].to_list()]
        close_prices = [float(v) for v in history[symbols[-1]].to_list()]

        # Calculate daily returns
        returns = [(close - open_) / open_ for open_, close in zip(open_prices, close_prices)]

        # Calculate mean and std of returns
        mean_return = sum(returns) / len(returns)
        std_return = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5

        # Adjust weights based on volatility
        weight_adjustment = min(self._scale_factor * std_return, 1.0)

        # Assign equal weights to all symbols
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
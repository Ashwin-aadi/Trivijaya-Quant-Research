from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy seeks to identify and follow long-term trends while scaling the "
        "positions based on recent volatility. High volatility suggests that momentum "
        "is likely, whereas low volatility may indicate a potential reversal."
    )

    def __init__(self, window: int = 20, scale_factor: float = 1.5) -> None:
        self._window = window
        self._scale_factor = scale_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol = history["symbol"].first()
        open_prices = [float(v) for v in history["open"].to_list()]
        close_prices = [float(v) for v in history["close"].drop_nulls().to_list()]

        if len(close_prices) < self._window:
            return Signal(information_available_at=stamp, weights={})

        returns = [(c / o - 1.0) for o, c in zip(open_prices[:-1], close_prices[1:])]
        volatility = abs(sum(returns)) / (len(returns) + 1e-8)

        if close_prices[-1] > max(close_prices[:-self._window]):
            weight = self._scale_factor * volatility
        elif close_prices[-1] < min(close_prices[:-self._window]):
            weight = -self._scale_factor * volatility
        else:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp,
            weights={symbol: weight}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
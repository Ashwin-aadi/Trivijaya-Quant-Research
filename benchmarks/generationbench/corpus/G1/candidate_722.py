from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends based on volatility. It identifies symbols with recent "
        "price movements that are above a certain threshold relative to their historical "
        "volatility. These symbols are then given higher weights in the portfolio."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_close = {symbol: float(v) for symbol, v in view.latest_close().items()}
        prices = [float(v) for _, v in history.to_dict().items() if v != "session_date"]

        volatility = _calculate_volatility(prices)
        recent_changes = [
            (latest_close[symbol] - open_price) / open_price
            for symbol, open_price in zip(view.symbols, prices[0])
        ]

        strong_trends: list[str] = []
        for symbol, change in zip(view.symbols, recent_changes):
            if change > self._threshold * volatility:
                strong_trends.append(symbol)

        weights = {s: 1.0 / len(strong_trends) for s in strong_trends}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_volatility(prices: list[float]) -> float:
    returns = [p / prices[i - 1] - 1.0 for i, p in enumerate(prices) if i > 0]
    volatility = (sum([r**2 for r in returns]) / len(returns)) ** 0.5
    return volatility
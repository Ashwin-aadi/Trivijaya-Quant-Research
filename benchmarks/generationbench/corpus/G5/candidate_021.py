from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "By tilting our portfolio towards low-volatility stocks, we aim to capture these "
        "outperformance benefits."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"]:
                continue
            prices = history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()
            if len(prices) < self._window:
                continue
            returns = [(prices[i + 1] - prices[i]) / prices[i] for i in range(len(prices) - 1)]
            volatility = (sum([r**2 for r in returns]) / len(returns)) ** 0.5
            volatilities[symbol] = float(volatility)

        sorted_symbols = [k for k, _ in sorted(volatilities.items(), key=lambda item: item[1])]
        picks = sorted_symbols[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 0.2 / len(picks)
        weights = {s: weight for s in picks}
        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
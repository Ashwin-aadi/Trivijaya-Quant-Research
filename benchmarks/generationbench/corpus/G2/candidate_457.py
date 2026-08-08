from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion occurs when asset prices tend to move back toward the long-term mean. "
        "In a short horizon, price deviations from this mean can be exploited for profit. "
        "If recent returns have been unusually large (positive or negative), they are likely to revert."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        recent_returns: list[float] = []
        for symbol in symbols:
            close_prices = history[symbol].to_list()
            if len(close_prices) < self._window:
                continue
            last_price = float(close_prices[-1])
            returns = [
                (last_price - close / last_price) / last_price
                for close in close_prices[:-1]
            ]
            recent_returns.extend(returns)

        if not recent_returns:
            return Signal(information_available_at=stamp, weights={})

        mean_return = sum(recent_returns) / len(recent_returns)
        above_mean = [r > mean_return for r in recent_returns]
        below_mean = [r < mean_return for r in recent_returns]

        buy_symbols = [symbols[i] for i, b in enumerate(above_mean) if not b][:5]
        sell_symbols = [symbols[i] for i, b in enumerate(below_mean) if not b][:5]

        weights = {s: 0.1 for s in buy_symbols}
        if buy_symbols:
            weights["cash"] = -sum(weights.values())

        return Signal(
            information_available_at=stamp, weights={**weights, **{s: 0.1 for s in sell_symbols}}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
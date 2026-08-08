from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "This can be attributed to the risk-return trade-off, where investors are willing to accept lower volatility in exchange for higher expected returns."
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

        volatilities: dict[str, float] = {}
        for symbol in symbols:
            close_prices = history[symbol].to_list()
            log_returns = [(close / close.shift(1) - 1.0) for close in close_prices[1:]]
            volatility = (sum(x**2 for x in log_returns) / len(log_returns)) ** 0.5
            volatilities[symbol] = volatility

        sorted_symbols = [symbol for symbol, _ in sorted(volatilities.items(), key=lambda item: item[1])]
        top_n_symbols = sorted_symbols[: min(self._window, len(sorted_symbols))]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
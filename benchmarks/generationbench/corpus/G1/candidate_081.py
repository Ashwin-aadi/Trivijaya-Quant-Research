from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility tilting is a strategy where we overweight stocks with lower historical "
        "volatility in the portfolio. This approach aims to capture alpha by exploiting the "
        "known empirical fact that low-volatility stocks tend to outperform the market over time."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_list = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbol_list:
            return Signal(information_available_at=stamp, weights={})

        volatilities: dict[str, float] = {}
        for symbol in symbol_list:
            values = [
                float(v) for v in history.select(pl.col(symbol)).to_series().to_list()
            ]
            volatility = _compute_volatility(values)
            volatilities[symbol] = volatility

        sorted_symbols = [k for k, v in sorted(volatilities.items(), key=lambda item: item[1])]
        top_symbols = sorted_symbols[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _compute_volatility(prices: list[float]) -> float:
    mean_price = sum(prices) / len(prices)
    variance = sum((p - mean_price) ** 2 for p in prices) / (len(prices) - 1)
    return variance**0.5


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_series().to_list()[0]
    assert isinstance(newest, date)
    return newest
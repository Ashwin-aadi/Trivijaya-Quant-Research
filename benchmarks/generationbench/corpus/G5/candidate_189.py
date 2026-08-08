from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility tilting is a strategy that aims to outperform the market by "
        "investing in stocks with lower historical volatility. The idea is that these "
        "stocks are less risky and may provide more stable returns over time."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(prices) < self._window + 1:
                continue
            returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
            volatility = (sum([r**2 for r in returns]) / (len(returns)-1)) ** 0.5
            volatilities[symbol] = volatility

        sorted_symbols = [k for k, v in sorted(volatilities.items(), key=lambda item: item[1])]
        top_n_symbols = sorted_symbols[:self._top_n]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Historically, low-volatility stocks have exhibited lower downside risk and higher "
        "expected returns. Tilting towards these assets can enhance portfolio performance."
    )

    def __init__(self, lookback: int = 60, top_n: int = 5) -> None:
        self._lookback = lookback
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if len(symbols) < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        volatilities = {}
        for symbol in symbols:
            close_prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(close_prices) <= 1:
                continue
            returns = [(close_prices[i] / close_prices[i - 1] - 1.0) for i in range(1, len(close_prices))]
            volatility = (sum([r**2 for r in returns]) / len(returns)) ** 0.5
            volatilities[symbol] = volatility

        sorted_symbols = [k for k, v in sorted(volatilities.items(), key=lambda item: item[1])]
        top_n_symbols = sorted_symbols[: self._top_n]
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
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often considered less risky and can provide stable returns. "
        "By tilting the portfolio towards these stocks, we aim to reduce overall portfolio risk."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        latest_closes = {symbol: float(v) for symbol, v in view.latest_close().items()}

        # Calculate historical volatility for each stock
        volatilities = {}
        for symbol in symbols:
            if symbol not in history.columns:
                continue

            prices = [float(p) for p in history[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue

            returns = [(prices[i] / prices[i - 1] - 1.0) for i in range(1, self._window)]
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
            volatility = (variance ** 0.5) * (self._window ** 0.5)

            volatilities[symbol] = volatility

        # Select top N low-volatility symbols
        sorted_symbols = [
            symbol for symbol, _ in sorted(volatilities.items(), key=lambda item: item[1])
        ][: self._top_n]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
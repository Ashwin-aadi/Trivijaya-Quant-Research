from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility tilting is based on the empirical observation that low-volatility stocks "
        "tend to outperform high-volatility stocks over time. This strategy selects the least volatile "
        "stocks from the NIFTY 100 index, focusing on minimizing portfolio risk while maintaining "
        "potential for returns."
    )

    def __init__(self, window: int = 30, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate daily returns
            returns = [
                (values[i] - values[i - 1]) / values[i - 1]
                for i in range(1, self._window)
            ]
            volatility = (sum(r**2 for r in returns)) ** 0.5 / (self._window - 1)

            volatilities[symbol] = volatility

        # Sort by volatility and pick the top N
        sorted_symbols = sorted(volatilities.items(), key=lambda x: x[1])
        picks = [symbol for symbol, _ in sorted_symbols[: self._top_n]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
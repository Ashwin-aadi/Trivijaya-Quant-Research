from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "This is often attributed to the risk premium investors demand for taking on additional risk."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatilities: list[float] = []
        symbols = set(history["symbol"].to_list())
        for symbol in symbols:
            prices = [float(v) for v in history.filter(pl.col("symbol") == symbol)[
                "adj_close"
            ].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue
            returns = [(prices[i] / prices[i - 1]) - 1.0 for i in range(1, len(prices))]
            volatility = (sum([r**2 for r in returns]) / len(returns)) ** 0.5
            volatilities.append(volatility)

        sorted_symbols = [s for _, s in sorted(zip(volatilities, symbols))]
        top_symbols = sorted_symbols[: int(0.3 * len(sorted_symbols))]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
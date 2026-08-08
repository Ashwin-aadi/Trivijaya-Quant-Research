from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the observation that high-volatility assets "
        "tend to exhibit stronger trends than low-volatility ones. By scaling the position size "
        "by historical volatility, we can leverage the asset's potential for sustained price movement."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history.select([pl.col("symbol"), pl.col("adj_close").alias("close")])

        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            returns = [(values[i + 1] - values[i]) / values[i] for i in range(len(values) - 1)]
            mean_return = sum(returns) / len(returns)
            volatility = (sum([r**2 for r in returns]) / len(returns)) ** 0.5
            volatilities[symbol] = volatility

        sorted_symbols = [s for s, v in sorted(volatilities.items(), key=lambda item: item[1], reverse=True)]
        
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_symbol = sorted_symbols[0]
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                top_symbol: weight * (volatilities[top_symbol] + 1),
                **{s: -weight for s in sorted_symbols if s != top_symbol},
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
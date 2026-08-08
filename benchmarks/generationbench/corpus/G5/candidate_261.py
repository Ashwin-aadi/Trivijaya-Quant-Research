from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility ones over the long term. "
        "By tilting towards low-volatility stocks, we aim to capture this outperformance."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        low_vol_symbols = self._select_low_volatility_stocks(history)

        weight = 1.0 / len(low_vol_symbols) if low_vol_symbols else 0
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in low_vol_symbols},
        )

    def _select_low_volatility_stocks(self, history: pl.DataFrame) -> list[str]:
        symbols = [str(s) for s in view.symbols if str(s) in history.columns]
        volatilities = []

        for symbol in symbols:
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            returns = [(adj_closes[i] - adj_closes[i - 1]) / adj_closes[i - 1] if i > 0 else 0.0 for i in range(len(adj_closes))]
            volatility = (sum(returns) ** 2) / self._window
            volatilities.append((symbol, volatility))

        sorted_volatilities = sorted(volatilities, key=lambda x: x[1])
        low_vol_symbols = [symbol for symbol, _ in sorted_volatilities[:5]]
        return low_vol_symbols


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
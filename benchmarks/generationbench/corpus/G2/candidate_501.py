from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over long periods. "
        "This is often attributed to the risk premium that investors demand for holding "
        "riskier assets. By tilting towards low-volatility stocks, we aim to capture this premium."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatilities: list[float] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, self._window)]
            volatility = (sum([r**2 for r in returns]) / self._window) ** 0.5
            volatilities.append(volatility)

        symbols_with_vol = {s: v for s, v in zip(view.symbols, volatilities)}
        sorted_symbols = [k for k, _ in sorted(symbols_with_vol.items(), key=lambda item: item[1])]
        
        weight = 1.0 / min(len(sorted_symbols), self._top_n)
        top_n_symbols = sorted_symbols[:self._top_n]
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
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform the market over the long term. By tilting "
        "our portfolio towards these stocks, we aim to capture this effect."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_list = [symbol for symbol in view.symbols if symbol in history.columns]
        history_filtered = history[symbol_list]

        volatilities: list[float] = []
        for symbol in symbol_list:
            adj_closes = history_filtered[f"{symbol}.adj_close"].to_list()
            returns = [(float(adj_closes[i]) - float(adj_closes[i-1])) / float(adj_closes[i-1])
                       if i > 0 else 0.0 for i in range(len(adj_closes))]
            volatility = (sum([r**2 for r in returns]) / len(returns)) ** 0.5
            volatilities.append(volatility)

        top_symbols = [symbol_list[i] for i, _ in
                       sorted(zip(range(len(symbol_list)), volatilities), key=lambda x: x[1])[:3]]

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
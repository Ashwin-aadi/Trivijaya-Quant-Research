from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy exploits the low-volatility anomaly in the Indian market by "
        "tilting the portfolio towards stocks with lower historical volatility. "
        "Risk-averse investor behavior and institutional constraints lead to underpricing of less volatile stocks, "
        "potentially offering higher risk-adjusted returns."
    )

    def __init__(self, window: int = 250) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        volatilities: dict[str, float] = {}
        for symbol in symbols:
            adj_closes = history[symbol].to_list()
            returns = [(adj_close - prev_adj_close) / prev_adj_close
                       for adj_close, prev_adj_close in zip(adj_closes[1:], adj_closes[:-1])]
            volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
            volatilities[symbol] = volatility

        sorted_volatilities = {k: v for k, v in sorted(volatilities.items(), key=lambda item: item[1])}
        num_symbols = len(symbols)
        low_vols = list(sorted_volatilities.keys())[:num_symbols // 4]
        weights = {s: 0.05 / len(low_vols) for s in low_vols} if low_vols else {}

        return Signal(
            information_available_at=stamp, weights={**weights}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
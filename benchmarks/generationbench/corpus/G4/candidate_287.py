from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ITSectorInflationStrategy(Strategy):
    rationale = (
        "This strategy combines sector-specific trends in the Indian IT sector with macroeconomic "
        "inflation indicators. It buys stocks in the IT sector when the sector is outperforming and "
        "the inflation rate is moderate, potentially exploiting arbitrage opportunities."
    )

    def __init__(self, it_symbols: tuple[str, ...] = ("TCS", "INFY", "WIPRO"), window: int = 30, inflation_threshold: float = 0.01) -> None:
        self._it_symbols = it_symbols
        self._window = window
        self._inflation_threshold = inflation_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        it_returns = []
        for symbol in self._it_symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            returns = [(adj_closes[i] / adj_closes[i - 1] - 1.0) for i in range(1, len(adj_closes))]
            it_returns.append(max(returns))

        inflation_rate = view.closes(lookback=30).select("session_date", "inflation_rate").sort("session_date").tail(1)["inflation_rate"].to_list()[0]
        if not 0.01 <= inflation_rate <= 0.04:
            return Signal(information_available_at=stamp, weights={})

        top_it_symbols = [symbol for symbol, r in zip(self._it_symbols, it_returns) if r > 0.02]
        weight_per_symbol = 1.0 / len(top_it_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_symbol for s in top_it_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
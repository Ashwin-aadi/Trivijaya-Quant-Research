from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "By tilting towards lower volatility stocks, we aim to capture this effect while mitigating risk."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(sym) for sym in view.symbols]
        closes = history.filter(pl.col("symbol").is_in(symbols)).select(
            pl.col("symbol"), pl.col("adj_close")
        )
        volatilities = _calculate_volatility(closes)
        sorted_symbols = [
            str(symbol) for symbol, volatility in sorted(volatilities.items(), key=lambda x: x[1])
        ]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_n = min(len(sorted_symbols), self._top_n)
        weight = 1.0 / top_n
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in sorted_symbols[:top_n]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_volatility(closes: pl.DataFrame) -> dict[str, float]:
    volatilities = {}
    for symbol in closes.unique(subset="symbol").iter_rows():
        symbol_closes = [float(v) for v in closes.filter(pl.col("symbol") == symbol[0])["adj_close"].to_list()]
        if len(symbol_closes) >= 2:
            returns = [(symbol_closes[i] - symbol_closes[i-1]) / symbol_closes[i-1] for i in range(1, len(symbol_closes))]
            volatilities[str(symbol)] = (sum([r**2 for r in returns])**0.5)
    return volatilities
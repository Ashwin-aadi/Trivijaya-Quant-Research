from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy selects stocks based on minimum average daily turnover and equal weights them "
        "to ensure a well-diversified and liquid portfolio. Rebalancing monthly ensures the portfolio remains "
        "responsive to market changes."
    )

    def __init__(self, min_turnover: float = 10_000_0000, top_n: int = 300) -> None:
        self._min_turnover = min_turnover
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=90)  # Look back for the past three months to calculate turnover
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if self._is_liquid_symbol(s, history)]
        if len(symbols) < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        weights = {symbol: 1.0 / self._top_n for symbol in symbols[:self._top_n]}
        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _is_liquid_symbol(symbol: str, history: pl.DataFrame) -> bool:
    avg_turnover = float(history.filter(pl.col("symbol") == symbol)["adj_close"].mean())
    return avg_turnover > 10_000_000
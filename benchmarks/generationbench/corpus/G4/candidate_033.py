from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy exploits the low-volatility effect by constructing a portfolio of "
        "stocks with lower historical volatility. Defensive stocks tend to outperform more "
        "volatile counterparts over time, making them attractive for risk-averse investors."
    )

    def __init__(self, window: int = 60, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history_with_returns = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=False)
            .drop_nulls()
        )

        # Calculate volatility for each stock
        volatilities = (
            history_with_returns.group_by("symbol")
            .agg((pl.col("return").std().alias("volatility")))
            .sort("volatility")
        )

        if volatilities.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [str(row[0]) for row in volatilities.head(self._top_n).to_dict(as_series=False).values()]

        # Assign equal weight to selected symbols
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
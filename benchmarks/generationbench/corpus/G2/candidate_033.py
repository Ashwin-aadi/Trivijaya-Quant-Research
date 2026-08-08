from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long run. "
        "This is often attributed to risk premium, where investors demand higher returns for "
        "taking on additional risk. By tilting towards low volatility, we can capture this "
        "outperformance."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = history.select(
            pl.col("session_date"),
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
        )

        # Compute rolling volatility for each symbol
        volatilities = (
            returns.lazy()
            .group_by("symbol")
            .agg((pl.col("return").std().over([pl.arange(1, self._window + 1)]).alias("volatility")))
            .collect()
        )

        # Sort by lowest volatility and pick top N symbols
        sorted_symbols = volatilities.sort("volatility", descending=False)["symbol"].to_list()[:5]
        
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long run. "
        "This is because low-volatility stocks are less risky and often have lower beta, making them attractive for risk-averse investors."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(s) for s in view.symbols]
        volatilities = history.select(
            pl.col("symbol").alias("symbol"),
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).mean().over("symbol").alias("volatility"),
        )[symbols]

        sorted_symbols = volatilities.sort("volatility", descending=False)["symbol"].to_list()[: len(symbols)]

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
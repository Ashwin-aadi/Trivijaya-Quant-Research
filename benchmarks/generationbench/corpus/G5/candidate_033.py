from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long run. "
        "By tilting our portfolio towards low-volatility stocks, we aim to capture this premium."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
        ).drop_nulls()

        # Group by symbol and calculate the standard deviation of daily returns
        std_devs = (
            history.group_by("symbol")
            .agg(pl.col("r").std().alias("volatility"))
            .sort(by="volatility", descending=False)
            .select(["symbol", "volatility"])
        )

        if std_devs.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in std_devs.to_dicts()[: self._top_n]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
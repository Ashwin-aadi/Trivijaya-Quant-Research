from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "By tilting towards low volatility, we aim to capture this effect."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window * 2:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the volatility for each stock
        volatility = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").std().alias("volatility")))
            .sort(by="volatility", descending=False)
        )

        if volatility.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Select the top N low-volatility stocks
        picks = [row["symbol"] for row in volatility.head(self._window)]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
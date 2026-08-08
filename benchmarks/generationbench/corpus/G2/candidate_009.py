from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over time. "
        "This is often attributed to the risk premium investors demand for taking on more "
        "risk. By tilting towards low-volatility stocks, we aim to capture this excess return."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date")
            .drop_nulls()
        )

        # Group by symbol and calculate the standard deviation of daily returns
        volatilities = (
            history.group_by("symbol")
            .agg(pl.col("r").std().alias("volatility"))
            .sort("volatility", descending=False)
            .select(["symbol", "volatility"])
        )

        if volatilities.height < 1:
            return Signal(information_available_at=stamp, weights={})

        # Select top low-volatility symbols
        picks = [row["symbol"] for row in volatilities.to_dicts()[:5]]

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
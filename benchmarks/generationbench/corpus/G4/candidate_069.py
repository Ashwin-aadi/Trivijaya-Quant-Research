from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility ones over time due to "
        "better risk-adjusted returns and reduced downside risk. This strategy exploits this "
        "trend by targeting low-volatility assets, aiming for higher risk-adjusted returns."
    )

    def __init__(self, window: int = 20, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily log returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date", descending=False)
            .drop_nulls(subset=["r"])
        )

        # Calculate 20-day standard deviation of daily returns for each stock
        volatilities = (
            history.groupby("symbol").agg(
                (pl.col("r").std().alias(f"volatility_{self._window}d"))
            ).sort(f"volatility_{self._window}d")
        )

        if volatilities.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        # Select top N low-volatility stocks
        picks = [str(row[0]) for row in volatilities.head(self._top_n).to_numpy()]
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
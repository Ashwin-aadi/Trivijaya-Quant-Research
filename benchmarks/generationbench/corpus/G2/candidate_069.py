from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over long periods. "
        "This effect is thought to be driven by risk aversion and the compensation investors "
        "require for taking on additional risk."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
        )
        history = history.sort("session_date")

        # Group by symbol and calculate the mean return over the lookback period
        mean_returns = history.group_by("symbol").agg(
            (pl.col("r").mean().alias("mean_return"))
        )

        # Sort symbols by mean return in descending order to get low volatility stocks at the top
        sorted_symbols = (
            mean_returns.sort("mean_return", descending=False)
            .select(["symbol"])
            .head(self._window)["symbol"]
            .to_list()
        )

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Allocate equal weight to the selected symbols
        weight = 1.0 / len(sorted_symbols)
        signal_weights = {s: weight for s in sorted_symbols}

        return Signal(
            information_available_at=stamp,
            weights=signal_weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
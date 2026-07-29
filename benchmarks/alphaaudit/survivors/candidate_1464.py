from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum by identifying symbols that have "
        "outperformed over the past 60 days. We expect these outperformers to continue their "
        "positive trend in the near future."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or len(history["symbol"].unique()) < 2:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
        )

        # Group by symbol and calculate mean return over the lookback period
        mean_returns = (
            history.group_by("symbol")
            .agg(pl.col("r").mean().alias("mean_return"))
            .sort("mean_return", descending=True)
        )

        if mean_returns.height < 2:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = mean_returns.head(5)["symbol"].to_list()

        # Assign equal weight to the top symbols
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
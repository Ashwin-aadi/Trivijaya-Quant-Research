"""Hold the most heavily traded half of the universe, equally weighted."""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible, top_n


class EqualWeightTopLiquidity(Strategy):
    """A liquidity tilt carrying no return forecast at all."""

    rationale = (
        "Restricting to the most traded names is what a large book must do regardless of any "
        "view, because thin names cannot absorb size. This isolates the return of that "
        "constraint on its own."
    )

    def __init__(self, lookback: int = 21, fraction: float = 0.5) -> None:
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be greater than zero and at most one")
        self._lookback = lookback
        self._fraction = fraction

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity = history.group_by("symbol").agg(
            pl.col("turnover_inr").median().alias("median_turnover")
        )
        scores = dict(
            zip(liquidity["symbol"].to_list(),
                liquidity["median_turnover"].to_list(), strict=True)
        )
        count = max(1, int(len(scores) * self._fraction))
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, count)),
        )

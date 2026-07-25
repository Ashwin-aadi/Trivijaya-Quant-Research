"""Rank on momentum scaled by how much money actually traded."""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible, top_n, window_return


class VolumeWeightedMomentum(Strategy):
    """Momentum, discounted where turnover is thin."""

    rationale = (
        "A price move on heavy turnover reflects more participants agreeing on a revaluation "
        "than the same move on a quiet day. Scaling momentum by traded value therefore favours "
        "trends with genuine weight behind them."
    )

    def __init__(self, lookback: int = 63, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        returns = window_return(view, self._lookback)
        history = view.history(lookback=self._lookback)
        if not returns or history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity = history.group_by("symbol").agg(
            pl.col("turnover_inr").median().alias("median_turnover")
        )
        turnover = dict(
            zip(liquidity["symbol"].to_list(),
                liquidity["median_turnover"].to_list(), strict=True)
        )
        largest = max(turnover.values()) if turnover else 0.0
        if largest <= 0:
            return Signal(information_available_at=stamp, weights={})

        # Turnover is normalised to the largest name, so the scale factor sits in [0, 1] and the
        # ranking stays driven by the return rather than by absolute size.
        scores = {
            symbol: value * (turnover.get(symbol, 0.0) / largest)
            for symbol, value in returns.items()
        }
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )

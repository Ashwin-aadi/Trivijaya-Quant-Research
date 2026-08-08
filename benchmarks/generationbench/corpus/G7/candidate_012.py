from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum3mo(Strategy):
    rationale = (
        "Cross-sectional momentum leverages the tendency for stocks with high recent "
        "performance to continue outperforming. By focusing on daily highs over a 3-month "
        "lookback period, we aim to capture these strong performers and allocate capital accordingly."
    )

    def __init__(self, lookback: int = 90, position_limit: float = 0.05, portfolio_limit: float = 0.03) -> None:
        self._lookback = lookback
        self._position_limit = position_limit
        self._portfolio_limit = portfolio_limit

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        high_prices = (
            history
            .select("symbol", pl.col("high"))
            .group_by("symbol")
            .agg(pl.col("high").max().alias("max_high"))
            .sort("max_high", descending=True)
            .head(20)["symbol"].to_list()
        )

        weights = {s: 1.0 / len(high_prices) for s in high_prices}
        if sum(weights.values()) > self._portfolio_limit:
            adjustment_factor = self._portfolio_limit / sum(weights.values())
            weights = {s: w * adjustment_factor for s, w in weights.items()}

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
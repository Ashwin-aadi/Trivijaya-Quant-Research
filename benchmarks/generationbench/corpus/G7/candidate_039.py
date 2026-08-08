from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeighted(Strategy):
    rationale = (
        "This strategy selects the top 50 liquid stocks based on daily volume over a 20-day lookback period. "
        "Each selected stock is given an equal weight in the portfolio to ensure liquidity-driven diversification."
    )

    def __init__(self, window: int = 20, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volume = (
            history.select(["symbol", "volume"])
            .group_by("symbol")
            .agg(pl.col("volume").sum().alias("total_volume"))
            .sort("total_volume", descending=True)
            .head(self._top_n)
            .select("symbol")
        )

        if volume.is_empty() or len(volume) < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / self._top_n
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in volume["symbol"].to_list()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
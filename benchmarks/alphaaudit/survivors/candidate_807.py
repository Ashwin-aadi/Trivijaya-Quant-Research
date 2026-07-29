from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity is a key driver of market efficiency. Assets with higher liquidity "
        "are less prone to price manipulation and are easier to trade without significant impact."
    )

    def __init__(self, lookback_days: int = 20) -> None:
        self._lookback_days = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.filter(pl.col("symbol").is_in(view.symbols))
            .group_by("symbol")
            .agg((pl.col("volume").mean().alias("mean_volume"),))
            .sort("mean_volume", descending=True)
            .head(self._lookback_days)
            .select("symbol", "mean_volume")
        )

        symbols = liquidity_scores["symbol"].to_list()
        weights = {s: 1.0 / len(symbols) for s in symbols}
        return Signal(
            information_available_at=stamp, weights={k: float(v) for k, v in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity-weighted equal weighting allocates capital to stocks based on their "
        "trading volume. This approach aims to distribute risk more evenly across highly liquid "
        "stocks, which are less likely to experience significant price volatility."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or len(view.symbols) == 0:
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.group_by("symbol")
            .agg(pl.col("volume").sum().alias("total_volume"))
            .sort("total_volume", descending=True)
            .head(len(view.symbols))
            .select(["symbol"])
            .to_dict(as_series=False)
        )

        weights = {k: 1 / len(liquidity_scores) for k in liquidity_scores.keys()}
        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
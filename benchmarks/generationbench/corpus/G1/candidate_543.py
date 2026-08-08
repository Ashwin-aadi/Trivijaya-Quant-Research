from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy screens for the most liquid stocks in terms of trading volume and "
        "equally weights them. Liquid stocks tend to have more reliable price signals and "
        "are less likely to be manipulated."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_volume_symbols = (
            history.lazy()
            .group_by("symbol")
            .agg(pl.col("volume").sum().alias("total_volume"))
            .sort("total_volume", descending=True)
            .head(10)
            .select(["symbol"])
            .collect()["symbol"]
            .to_list()
        )

        if not high_volume_symbols:
            return Signal(information_available_at=stamp, weights={})

        equal_weight = 1.0 / len(high_volume_symbols)
        return Signal(
            information_available_at=stamp,
            weights=dict(zip(high_volume_symbols, [equal_weight] * len(high_volume_symbols))),
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
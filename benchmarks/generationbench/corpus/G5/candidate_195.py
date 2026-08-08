from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "By screening stocks with sufficient trading volume, we ensure that only liquid "
        "assets are considered for the portfolio. Equal weighting across selected stocks avoids "
        "concentration risk and leverages the diversification benefits of a well-diversified market index."
    )

    def __init__(self, min_volume: int = 100000) -> None:
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=252)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_filter = (
            history.group_by("symbol")
            .agg(pl.col("volume").sum().alias("total_volume"))
            .filter(pl.col("total_volume") > self._min_volume)
            .select(pl.col("symbol"))
        )

        if volume_filter.height < 5:
            return Signal(information_available_at=stamp, weights={})

        selected_symbols = [str(s) for s in volume_filter["symbol"].to_list()]
        equal_weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: equal_weight for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
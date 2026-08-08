from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy selects the most liquid stocks based on recent trading volume. "
        "By equal-weighting these stocks, we aim to benefit from higher liquidity while maintaining a balanced portfolio."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the average daily volume over the lookback period
        volume_df = history.group_by("symbol").agg(
            pl.col("volume").mean().alias("avg_volume")
        )
        sorted_symbols = volume_df.sort("avg_volume", descending=True).select(
            "symbol"
        ).to_series().to_list()

        if len(sorted_symbols) < 5:
            return Signal(information_available_at=stamp, weights={})

        top_liquids = sorted_symbols[:5]
        weight = 1.0 / len(top_liquids)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_liquids},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
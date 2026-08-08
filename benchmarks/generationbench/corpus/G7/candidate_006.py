from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeBasedRelativeStrength(Strategy):
    rationale = (
        "This strategy selects stocks based on their relative strength in terms of average "
        "trading volume over a 20-day period. Higher trading volumes indicate greater interest "
        "and liquidity, which can signal more sustainable trends."
    )

    def __init__(self, window: int = 20, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_avg = (
            history.group_by("symbol")
            .agg((pl.col("volume").sum() / self._window).alias("avg_volume"))
            .sort("avg_volume", descending=True)
        )
        top_stocks = volume_avg["symbol"].to_list()[: self._top_n]

        if len(top_stocks) < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        weight_step = 100.0 / (self._top_n - 1)
        weights = {s: max(2.0, 100.0 - i * weight_step) for i, s in enumerate(top_stocks)}

        return Signal(
            information_available_at=stamp,
            weights={s: w / 100.0 for s, w in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
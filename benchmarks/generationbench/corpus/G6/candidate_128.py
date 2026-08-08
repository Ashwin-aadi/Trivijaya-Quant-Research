from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMoves(Strategy):
    rationale = (
        "This strategy leverages strong price trends confirmed by high trading volumes to "
        "capitalize on broad market participation and commitment. It identifies directional "
        "moves based on closing prices outside the previous day's range with volume above a 20-day average, ensuring robust trend identification."
    )

    def __init__(self, window: int = 20, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        volume_avg = closes.select(pl.col("adj_close").shift(-1) > pl.col("adj_close").mean().alias("above_mean")).group_by("symbol").agg(
            (pl.col("above_mean") & (closes["volume"] > history.select(pl.col("volume").mean()).to_series())).sum().alias("condition")
        ).sort("condition", descending=True).select(pl.col("symbol").head(self._top_n)).to_dict(False)

        if not volume_avg:
            return Signal(information_available_at=stamp, weights={})

        weight = 0.05
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [item[0] for item in volume_avg.values()]}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
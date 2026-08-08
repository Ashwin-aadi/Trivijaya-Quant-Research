from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy exploits the economic mechanism where smaller-cap stocks often exhibit "
        "higher abnormal returns due to their low liquidity. By screening for equities based on "
        "liquidity metrics and then equally weighting the selected stocks, we aim to mitigate "
        "concentration risk and capitalize on potential higher returns from underfollowed "
        "smaller-cap stocks."
    )

    def __init__(self, window: int = 20, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_history = (
            history.select(
                pl.col("symbol"), (pl.col("volume") / 20).alias("avg_volume")
            )
            .group_by("symbol")
            .agg(pl.col("avg_volume").mean().alias("average_volume"))
            .sort("average_volume", descending=False)
            .head(self._top_n)
        )

        picks = volume_history.select("symbol").to_dict(False)
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
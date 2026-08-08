from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy exploits the phenomenon where smaller-cap stocks with higher liquidity "
        "exhibit better price efficiency and reduced transaction costs. By selecting stocks based "
        "on their daily trading volume, we aim to capture superior risk-adjusted returns."
    )

    def __init__(self, min_volume: int = 1_000_000, top_n: int = 30) -> None:
        self._min_volume = min_volume
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=None)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_df = (
            history.lazy()
            .select(
                pl.col("symbol"),
                (pl.col("volume") / pl.col("session_date").cast(pl.Date).dt.days_in_month()).alias("avg_daily_volume"),
            )
            .filter(pl.col("avg_daily_volume") > self._min_volume)
            .group_by("symbol")
            .agg(pl.col("avg_daily_volume").mean().alias("avg_daily_volume"))
            .sort("avg_daily_volume", descending=True)
            .head(self._top_n)
        ).collect()

        if volume_df.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        picks = [row["symbol"] for row in volume_df.to_dicts()]
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
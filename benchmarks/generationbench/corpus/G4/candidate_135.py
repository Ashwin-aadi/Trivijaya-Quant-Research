from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy focuses on highly liquid stocks to capitalize on the liquidity premium. "
        "By equally weighting selected equities based on their liquidity metrics, we aim to benefit "
        "from lower trading costs and higher price accuracy, which are often associated with larger "
        "and more frequently traded stocks."
    )

    def __init__(self, window_volume: int = 20, window_turnover: int = 30, top_n: int = 15) -> None:
        self._window_volume = window_volume
        self._window_turnover = window_turnover
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._window_volume, self._window_turnover))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_data = (
            history.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("volume").sum()).alias("total_volume"),
                pl.col("close").mean().alias("average_close"),
            )
            .collect()
        )

        turnover_data = (
            view.closes(lookback=self._window_turnover)
            .lazy()
            .join(volume_data, on="symbol")
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("average_close") - 1.0).alias("turnover_ratio"),
            )
        )

        liquidity_scores = (
            turnover_data.with_columns(
                ((pl.col("total_volume").rank(method="max", descending=True)) / 20).alias("volume_rank"),
                (pl.col("turnover_ratio").rank(method="max", descending=True)) / 30
            ).select(
                pl.col("symbol"),
                "volume_rank",
                "turnover_ratio"
            )
        )

        liquidity_scores = (
            liquidity_scores.sort(["volume_rank", "turnover_ratio"], descending=[True, True])
            .head(self._top_n)
            .with_columns((1 / self._top_n).alias("weight"))
        )

        weights = {row["symbol"]: row["weight"] for row in liquidity_scores.iter_rows()}
        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
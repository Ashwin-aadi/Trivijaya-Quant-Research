from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion occurs when prices return to the average value over a period. "
        "In this strategy, we identify stocks that have deviated significantly from their mean price "
        "and are expected to revert to it, providing potential trading opportunities."
    )

    def __init__(self, window: int = 50, z_score_threshold: float = 2.0) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history["symbol"].to_list()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.groupby("symbol").agg(
            (pl.col("adj_close").mean()).alias("mean")
        )
        z_score = (
            view.closes(lookback=self._window)
            .with_columns((pl.col("adj_close") - pl.col("mean")).alias("diff"))
            .with_columns(((pl.col("diff") / pl.col("mean")).abs().rank(
                method="dense", descending=True
            )).alias("z_score"))
        )

        high_z_score = z_score.filter(pl.col("z_score") > self._z_score_threshold)
        low_z_score = z_score.filter(pl.col("z_score") < -self._z_score_threshold)

        picks: list[str] = []
        if not high_z_score.is_empty():
            picks.extend(high_z_score["symbol"].to_list())
        if not low_z_score.is_empty():
            picks.extend(low_z_score["symbol"].to_list())

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
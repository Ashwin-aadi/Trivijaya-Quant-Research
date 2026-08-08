from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion occurs when recent extreme prices are followed by prices moving "
        "towards the mean. By identifying symbols that have recently deviated from their "
        "mean and are close to reverting, we can profit from this behavior."
    )

    def __init__(self, window: int = 10, z_score_threshold: float = 2.0) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        means = (
            history.group_by("symbol")
            .agg(
                (pl.col("adj_close").mean().alias("mean")),
                (pl.col("adj_close").std().alias("std")),
            )
            .with_columns((pl.col("adj_close") - pl.col("mean")).alias("deviation"))
        )

        recent_closes = view.closes(lookback=self._window)
        z_scores = (
            means.join(
                right=recent_closes,
                on="symbol",
                how="left",
            )
            .with_columns((pl.col("adj_close") - pl.col("mean")) / pl.col("std").alias("z_score"))
            .select(["symbol", "z_score"])
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in z_scores.columns or z_scores[z_scores["symbol"] == symbol].height < 1:
                continue
            if float(z_scores[z_scores["symbol"] == symbol]["z_score"].to_list()[0]) >= self._z_score_threshold:
                picks.append(symbol)

        if not picks:
            return Signal(information_available_at=stamp, weights={})
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
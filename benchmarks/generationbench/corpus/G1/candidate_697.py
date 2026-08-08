from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion suggests that prices which deviate significantly from "
        "their recent average will revert to that average. This can be exploited for profit."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        means = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean().alias("mean")))
            .with_column(
                (pl.col("adj_close") / pl.col("mean") - 1.0).alias("z_score"),
            )
        )

        if means.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        z_scores = means.select("z_score").to_dict(as_series=False)
        z_scores = {k: v for k, v in z_scores.items() if len(v) == self._window}
        candidates = [
            symbol
            for symbol, scores in z_scores.items()
            if any(score < -1.0 or score > 1.0 for score in scores[-5:])
        ]

        weight = 1.0 / len(candidates)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in candidates},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
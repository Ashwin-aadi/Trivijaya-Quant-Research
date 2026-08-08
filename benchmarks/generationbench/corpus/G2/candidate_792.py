from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "relative to their historical levels in recent periods to continue performing well. This "
        "strategy aims to identify such stocks and allocate capital accordingly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("momentum_score"),
                (pl.col("session_date")).max().alias("latest_date")
            )
        )

        momentum_scores = (
            momentum_scores
            .filter(pl.col("latest_date") == stamp)
            .sort("momentum_score", descending=True)
            .head(5)
        )

        if momentum_scores.height < 1:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / momentum_scores.height
        symbols = [row["symbol"] for row in momentum_scores.iter_rows()]
        weights = {s: weight for s in symbols}
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
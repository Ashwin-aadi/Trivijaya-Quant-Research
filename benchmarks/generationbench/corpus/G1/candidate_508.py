from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion(Strategy):
    rationale = (
        "Price reversion strategies exploit the tendency of stock prices to revert "
        "to a mean level over time. This strategy identifies stocks that have "
        "deviated significantly from their trailing average and bets on a return "
        "to equilibrium."
    )

    def __init__(self, window: int = 20, z_score_threshold: float = 2.0) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < 2 * self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols_with_data = [symbol for symbol in view.symbols if symbol in history.columns]
        filtered_history = history[symbols_with_data]

        z_scores = (
            filtered_history
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") - pl.col("adj_close").mean().over("symbol")) /
                pl.col("adj_close").std().over("symbol")
            )
            .with_columns(pl.col("adj_close").mean().over("symbol").alias("trailing_mean"))
        )

        reversion_candidates = (
            z_scores
            .filter(
                (pl.col("adj_close") < -self._z_score_threshold * pl.col("std")) |
                (pl.col("adj_close") > self._z_score_threshold * pl.col("std"))
            )
            .select(["symbol", "trailing_mean"])
        )

        if reversion_candidates.is_empty():
            return Signal(information_available_at=stamp, weights={})

        weights = {row["symbol"]: 1.0 / len(reversion_candidates) for row in reversion_candidates.iter_rows()}
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
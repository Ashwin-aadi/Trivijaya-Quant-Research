from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion50d(Strategy):
    rationale = (
        "This strategy identifies stocks that have deviated significantly from their 50-day "
        "Simple Moving Average (SMA) and are expected to revert. It combines elements of Design 1 for "
        "simplicity and broader applicability with the emphasis on equal weighting from Design 3."
    )

    def __init__(self, window: int = 50, z_score_threshold_entry: float = -1.5, 
                 z_score_threshold_exit: float = 1.2, holding_period: int = 30) -> None:
        self._window = window
        self._z_score_threshold_entry = z_score_threshold_entry
        self._z_score_threshold_exit = z_score_threshold_exit
        self._holding_period = holding_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes().with_column(pl.col("session_date").alias("date"))

        # Calculate the 50-day SMA
        sma = (
            history.select(pl.col("adj_close"))
                   .rolling_mean(self._window)
                   .last()
                   .to_series()
        )

        # Compute z-scores for each symbol
        sma_closes = closes.join(history, on="symbol", how="left")
        sma_closes = sma_closes.with_columns(
            (pl.col("adj_close") - pl.col(f"close.{self._window}")) /
            (pl.col("adj_close").std().over(f"symbol.{self._window}") + 1e-6).alias("z_score")
        )

        # Identify entries
        candidates = sma_closes.filter(
            (pl.col("date") == view.as_of) &
            (pl.col("z_score") < self._z_score_threshold_entry)
        )["symbol"].to_list()

        if not candidates:
            return Signal(information_available_at=stamp, weights={})

        # Exit criteria
        exits = (
            sma_closes.filter(
                (pl.col("date") == view.as_of) &
                (pl.col("z_score") > self._z_score_threshold_exit)
            )["symbol"].to_list() +
            [c for c in candidates if (view.history().filter(pl.col("symbol") == c).height >= self._holding_period)]
        )

        # Determine final weights
        picks = list(set(candidates) - set(exits))
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Large volume moves are often associated with significant market events or news. "
        "If a stock's price direction is confirmed by a high trading volume, it suggests "
        "strong institutional involvement and potentially higher future returns."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        # Calculate price changes and volumes
        price_changes = (
            history.with_columns(
                (pl.col("adj_close").shift(-1) - pl.col("adj_close")).alias("price_change"),
                (pl.col("volume") / pl.col("volume").shift(1)).alias("volatility"),
            )
            .sort("session_date")
            .filter(pl.col("session_date") != view.as_of)
        )

        # Identify breakout candidates
        breakout_candidates = price_changes.filter(
            (pl.col("price_change").abs() >= 0.05) & (pl.col("volatility") > 1.2)
        ).select(pl.col("symbol"))

        if breakout_candidates.height == 0:
            return Signal(information_available_at=stamp, weights={})

        # Assign equal weight to each candidate
        top_n = min(breakout_candidates.height, self._window)
        symbol_weights = {s: 1.0 / top_n for s in breakout_candidates.to_series().head(top_n)}

        return Signal(
            information_available_at=stamp,
            weights=symbol_weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
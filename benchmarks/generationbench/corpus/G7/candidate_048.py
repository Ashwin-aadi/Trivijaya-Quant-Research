from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "relative to their peers in the recent past to continue outperforming. By ranking "
        "stocks based on their returns over 60 days and selecting top performers, we aim to "
        "capitalize on these trends."
    )

    def __init__(self, window: int = 60, positions: int = 30) -> None:
        self._window = window
        self._positions = positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .drop("open", "high", "low", "volume")
        )

        # Compute momentum scores
        momentum_scores = (
            history.group_by("symbol")
            .agg(
                pl.col("return").mean().alias("momentum_score"),
            )
            .sort("momentum_score", descending=True)
            .head(self._positions)[["symbol"]]
        )

        weights = {s: 1.0 / self._positions for s in momentum_scores["symbol"].to_list()}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max())["session_date"][0]
    assert isinstance(newest, date)
    return newest
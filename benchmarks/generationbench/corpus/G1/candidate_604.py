from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Price reversion occurs when the current price moves back towards a historical mean. "
        "This strategy identifies stocks where prices have deviated significantly from their "
        "trailing mean and are likely to revert to it."
    )

    def __init__(self, window: int = 20, z_score_threshold: float = 2.0) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or not all(symbol in history.columns for symbol in view.symbols):
            return Signal(information_available_at=stamp, weights={})

        mean_adj_close = history.group_by("symbol").agg(pl.col("adj_close").mean().alias("mean")).collect()
        std_adj_close = history.group_by("symbol").agg(pl.col("adj_close").std().alias("std")).collect()

        adj_closes = view.closes(lookback=self._window)
        z_scores = (adj_closes - mean_adj_close["mean"].to_list()) / std_adj_close["std"].to_list()
        
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in z_scores.columns or pl.col(symbol).is_null().any():
                continue
            z_score = float(z_scores[symbol][-1])
            if abs(z_score) >= self._z_score_threshold and history.select(pl.col("adj_close").filter(pl.col("symbol") == symbol).last()).collect()[0, 0] != adj_closes[symbol].to_list()[-1]:
                picks.append(symbol)

        picks = picks[:5]
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
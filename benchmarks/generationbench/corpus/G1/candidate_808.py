from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingAvg(Strategy):
    rationale = (
        "Reversion to the mean suggests that prices which are extreme in one period "
        "are likely to revert towards their long-term average. This strategy identifies "
        "overbought or oversold conditions and bets on reversion."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        avg_close = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("avg_close"))
            .with_columns(
                (pl.col("adj_close") - pl.col("avg_close")).abs().alias("deviation")
            )
        )

        closest_to_avg: dict[str, float] = (
            avg_close.sort("deviation", descending=True)
            .select(["symbol"])
            .head(5)["symbol"]
            .to_list()
        )[:3]

        if not closest_to_avg:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(closest_to_avg)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in closest_to_avg},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
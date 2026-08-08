from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy selects stocks based on their recent high price performance. "
        "Higher highs indicate stronger intraday momentum and potential for future gains."
    )

    def __init__(self, window: int = 30, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily high returns
        history = (
            history.sort("session_date")
            .with_columns(
                (pl.col("high") / pl.col("high").shift(1) - 1.0).alias("return")
            )
            .select(["symbol", "session_date", "return"])
        )

        # Select top N symbols with highest average returns
        avg_returns = (
            history.group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
            .sort("avg_return", descending=True)
            .head(self._top_n)
        )

        picks: list[str] = [row["symbol"] for row in avg_returns.to_dicts()]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
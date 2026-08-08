from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy invests in the top-performing stocks based on their "
        "recent returns. The idea is that past winners are likely to continue outperforming."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history_with_returns = (
            history
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
        )

        top_performers = (
            history_with_returns
            .group_by("symbol")
            .agg(
                (pl.col("return").mean().alias("avg_return")),
                pl.col("adj_close").last().alias("latest_close"),
            )
            .sort("avg_return", descending=True)
            .select(["symbol", "avg_return"])
        )

        if top_performers.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        picks = [row["symbol"] for row in top_performers.rows()[:self._top_n]]
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
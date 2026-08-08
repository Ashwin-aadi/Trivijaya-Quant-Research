from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum suggests that securities with higher returns in the recent "
        "past will continue to outperform. This strategy exploits this phenomenon by investing "
        "in the top performing stocks over a lookback period."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width == 0:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            (closes.with_columns((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns"))
              .select(pl.exclude("symbol", "session_date")))
        )

        # Handle division by zero and NaN values
        returns["returns"] = returns["returns"].fill_null(0)

        # Find the top performing symbols by summing their daily returns
        ranked_symbols = (
            returns.select(
                pl.sum("returns").over("symbol")
                .rank(method="dense", descending=True)
                .alias("rank")
            )
            .select(["symbol", "rank"])
            .filter(pl.col("rank") <= self._top_n)
        )

        # Extract the top performing symbols
        picks = ranked_symbols["symbol"].to_list()

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights=dict(zip(picks, [weight] * len(picks)))
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
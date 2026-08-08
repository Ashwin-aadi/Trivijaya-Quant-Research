from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have outperformed "
        "their peers in recent history to continue outperforming. This strategy ranks stocks "
        "by their recent returns and allocates capital to the top performers."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < (self._window * len(view.symbols)):
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        ).sort("session_date")

        # Calculate cumulative returns over the window
        history = history.with_columns(
            (
                pl.col("return")
                .rank(method="dense", descending=True)
                .over("symbol")
                .alias("cumulative_return_rank")
            )
        )

        # Get top N performing symbols based on cumulative return rank
        picks: list[str] = []
        for symbol in view.symbols:
            if (symbol := history.filter(pl.col("symbol") == symbol).height) < 1:
                continue
            rank_series = history.filter(pl.col("symbol") == symbol)["cumulative_return_rank"]
            top_rank = rank_series.min().to_list()[0]
            if top_rank <= self._top_n:
                picks.append(symbol)

        picks = picks[: self._top_n]
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
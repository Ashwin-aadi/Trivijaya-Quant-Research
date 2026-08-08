from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum identifies stocks that have performed well recently and "
        "allocates capital towards these winners to capture continued outperformance."
    )

    def __init__(self, window: int = 20, num_top_stocks: int = 5) -> None:
        self._window = window
        self._num_top_stocks = num_top_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        returns = (
            closes
            .lazy()
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
            .sort("avg_return", descending=True)
            .select(["symbol", "avg_return"])
        ).collect()

        top_stocks = [row["symbol"] for row in returns.to_dicts()[: self._num_top_stocks]]
        if not top_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_stocks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
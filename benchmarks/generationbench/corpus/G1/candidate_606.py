from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with a higher relative strength indicate they have outperformed the broader market "
        "recently. This suggests they may continue to outperform due to favorable momentum or other factors."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
        ).sort("session_date")

        # Calculate the average return of each stock over the lookback period
        avg_returns = (
            history.groupby("symbol")
                   .agg(
                       (pl.col("returns").mean()).alias("avg_return"),
                       (pl.col("adj_close").last()).alias("latest_close"),
                   )
                   .sort("avg_return", descending=True)
        )

        # Get the top N symbols based on their average return
        top_symbols = avg_returns["symbol"].to_list()[:self._top_n]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
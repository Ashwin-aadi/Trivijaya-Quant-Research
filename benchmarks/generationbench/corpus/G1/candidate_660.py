from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Identify stocks with the strongest recent price momentum by ranking them and "
        "allocating capital to the top performers."
    )

    def __init__(self, window: int = 20, num_top_stocks: int = 5) -> None:
        self._window = window
        self._num_top_stocks = num_top_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .with_column(pl.col("return").rank(method="dense", descending=True).alias("momentum_rank"))
        )

        # Get the top N stocks by momentum rank
        top_stocks = history.select(
            pl.col("symbol"),
            pl.col("momentum_rank")
        ).top_n(self._num_top_stocks, by="momentum_rank").to_dict(True)

        if not top_stocks["symbol"]:
            return Signal(information_available_at=stamp, weights={})

        # Assign equal weight to the top stocks
        weight = 1.0 / len(top_stocks["symbol"])
        weights = {s: weight for s in top_stocks["symbol"]}

        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
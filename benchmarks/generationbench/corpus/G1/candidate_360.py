from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency of stocks that have outperformed in "
        "the recent past to continue outperforming. This strategy allocates capital to top "
        "performers over a specified lookback period."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * 2 + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns
        returns = (
            closes.select(pl.col("adj_close").to_list())
                   .shift(-1)
                   .lazy()
                   .with_columns(
                       (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
                   )
                   .collect()
        )

        # Filter out null values
        returns = returns.select(pl.col("symbol", "return")).drop_nulls()

        # Rank symbols by return and select top performers
        ranked = (
            returns.group_by("symbol").agg(
                (pl.col("return").mean().alias("avg_return"))
            )
                   .sort("avg_return", descending=True)
                   .head(self._top_n + 1)  # Add one to handle ties or extra entries
        )

        symbols = ranked["symbol"].to_list()
        weights = {s: 1.0 / len(symbols) for s in symbols}

        return Signal(
            information_available_at=stamp,
            weights={s: weights.get(s, 0.0) for s in view.symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
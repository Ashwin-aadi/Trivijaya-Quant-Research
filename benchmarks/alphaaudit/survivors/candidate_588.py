from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy identifies stocks with the highest momentum based on "
        "recent price appreciation. These stocks are expected to continue outperforming."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.latest_close()

        # Calculate returns for each stock
        returns = (
            history.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("r"),
                (pl.col("session_date")).max().alias("latest_date"),
            )
            .collect()
        )

        # Filter out symbols not in the latest closes
        filtered_returns = returns.filter(
            pl.col("symbol").is_in(latest_closes.keys())
        )

        # Find top N performers based on recent returns
        top_symbols = (
            filtered_returns.sort("r", descending=True)
            .head(self._top_n)
            .select(["symbol"])
            .to_series()
            .to_list()
        )

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
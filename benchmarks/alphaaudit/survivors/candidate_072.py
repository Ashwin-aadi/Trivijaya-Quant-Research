from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed "
        "well in recent periods to continue outperforming. By focusing on top performers, "
        "we can generate positive returns."
    )

    def __init__(self, window: int = 20, num_top_stocks: int = 5) -> None:
        self._window = window
        self._num_top_stocks = num_top_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .head(self._window)
        )

        # Rank symbols by return
        ranked = (
            history.group_by("symbol")
            .agg(
                pl.col("return").mean().alias("average_return"),
            )
            .sort("average_return", descending=True)
        )

        top_stocks = ranked["symbol"].to_list()[: self._num_top_stocks]

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
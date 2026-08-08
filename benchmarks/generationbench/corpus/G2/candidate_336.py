from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have outperformed "
        "in the recent past to continue to outperform. This strategy ranks stocks based on their "
        "returns over a short period and invests in the top performers."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns for each stock
        returns = (closes["adj_close"] / closes["adj_close"].shift(self._window) - 1.0).alias("r")

        # Get the most recent close prices and returns
        latest_closes = view.closes().with_column(returns)
        if latest_closes.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Rank stocks by their returns in descending order
        ranked = (
            latest_closes.sort("r", descending=True)
            .group_by("symbol")
            .agg((pl.col("adj_close").last()).alias("latest_close"))
        )

        # Select the top N performers
        top_performers = ranked.head(self._window)["symbol"].to_list()

        if not top_performers:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_performers)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_performers},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
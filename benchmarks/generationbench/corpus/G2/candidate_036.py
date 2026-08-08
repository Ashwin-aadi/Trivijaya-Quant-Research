from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have outperformed "
        "their peers in recent periods to continue outperforming. This strategy selects the top "
        "performers based on their returns over a short window and allocates capital accordingly."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate returns for each symbol
        returns = (
            closes.melt()
            .filter(pl.col("symbol") != "session_date")
            .group_by("symbol")
            .agg((pl.col("value").shift(-1) / pl.col("value").first() - 1.0).alias("return"))
            .sort("return", descending=True)
        )

        # Get top performing symbols
        top_performers = returns.head(self._window)["symbol"].to_list()

        if not top_performers:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_performers)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_performers}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
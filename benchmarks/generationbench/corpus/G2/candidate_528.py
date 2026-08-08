from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed "
        "well relative to their peers in the recent past to continue outperforming. This "
        "strategy seeks to invest in the top-performing stocks over a short-term window."
    )

    def __init__(self, lookback: int = 20) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)

        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the percentage change in price over the lookback period
        changes = (closes["adj_close"] / closes["adj_close"].shift(self._lookback) - 1.0).alias("change")
        top_performers = (
            closes.with_columns(changes)
            .sort("change", descending=True)
            .select(pl.col("symbol"))
            .head(5)["symbol"]
            .to_list()
        )

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
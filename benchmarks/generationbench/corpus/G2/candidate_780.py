from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum identifies stocks that have outperformed their peers in the "
        "recent past. Such stocks are likely to continue performing well due to their strong "
        "momentum and investor sentiment."
    )

    def __init__(self, lookback: int = 60) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)

        if closes.height < self._lookback or closes.width < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns for the lookback period
        returns_df = (
            closes.sort("session_date")
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._lookback) - 1.0).alias("return")
            )
        )

        # Identify top performing stocks
        ranked_returns = returns_df.with_columns(
            (pl.col("return").rank(method="dense", descending=True)).alias("rank")
        ).sort("symbol")

        top_performers = ranked_returns.filter(pl.col("rank") <= 5)["symbol"].to_list()

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
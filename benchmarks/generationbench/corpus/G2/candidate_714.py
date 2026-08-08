from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks that outperform the broad market tend to continue outperforming in the near "
        "term. This suggests a relative strength strategy can generate alpha by holding the "
        "strongest performers."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

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
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
        )

        # Get the top performers
        top_performers = history.select(
            pl.col("symbol"), pl.col("avg_return").rank(method="dense", descending=True)
        ).filter(pl.col("avg_return").rank(method="dense", descending=True) <= 5)

        if top_performers.height < 5:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row[0] for row in top_performers.iter_rows()]
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
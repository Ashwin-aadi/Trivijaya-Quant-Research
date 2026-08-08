from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "The relative strength strategy selects the top-performing stocks based on their "
        "performance against the broader market. This assumes that strong performers are more likely to continue outperforming."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date", descending=False)
            .group_by("symbol")
            .agg(pl.col("r").sum().alias("total_return"))
        )

        # Handle missing data and consecutive identical prices
        history = (
            history.with_columns(
                (pl.col("r") != 0).alias("non_zero_return"),
                pl.col("r").shift(1) != pl.col("r").shift(-1).alias("consecutive_same"),
            )
            .filter((pl.col("non_zero_return")) & (~pl.col("consecutive_same")))
            .with_columns(
                pl.col("total_return").rank(method="dense", descending=True, ties_method="dense").alias("rank")
            )
        )

        # Rank by total return
        ranked = history.sort("rank").limit(self._top_n)
        
        top_symbols = [row["symbol"] for row in ranked.to_dicts()]
        
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
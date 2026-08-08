from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the idea that stocks in an index that have "
        "recently outperformed their peers are likely to continue outperforming. This is a "
        "widely observed phenomenon and can be attributed to investor behavior, market efficiency, "
        "and the persistence of performance."
    )

    def __init__(self, lookback_window: int = 60, top_n: int = 10) -> None:
        self._lookback_window = lookback_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the percentage change in price for each stock
        price_changes = (
            history.group_by("symbol")
                   .agg((pl.col("adj_close") / pl.col("adj_close").shift(self._lookback_window) - 1.0).alias("return"))
                   .sort("return", descending=True)
                   .head(self._top_n)
        )

        top_symbols = [row["symbol"] for row in price_changes.to_dicts()]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum identifies stocks that have outperformed the market "
        "over a recent period and allocates capital to them in an attempt to capture future "
        "performance."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns
        prices = history["adj_close"]
        price_changes = (prices / prices.shift(1) - 1.0).alias("return")
        returns_df = history.with_columns(price_changes)

        # Rank by return
        ranked_returns = returns_df.select(
            pl.col("symbol"),
            pl.col("return").rank(method="max", descending=True).alias("rank"),
        )

        top_symbols = ranked_returns.sort("rank").select(pl.col("symbol")).head(5)["symbol"].to_list()

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
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength compared to the broader market "
        "tend to outperform. This is based on the assumption that stocks "
        "that have performed well relative to their peers are more likely "
        "to continue this positive trend."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate returns
        returns = (
            (closes.lazy()
             .with_column((pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return"))
             .group_by("symbol")
             .agg(pl.col("return").mean().alias("avg_return"))
             .sort("avg_return", descending=True)
             .collect()
            )["avg_return"]
        ).to_list()

        # Ensure we have enough symbols to rank
        if len(returns) < 1:
            return Signal(information_available_at=stamp, weights={})

        top_n = min(len(returns), view.symbols.__len__())
        top_symbols = [symbol for symbol in view.symbols]
        
        weight = 1.0 / top_n
        return Signal(
            information_available_at=stamp,
            weights=dict(zip(top_symbols, [weight] * top_n)),
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
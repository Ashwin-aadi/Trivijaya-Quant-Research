from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion suggests that stocks which are far from their "
        "short-term average price will eventually revert to it. This strategy aims to identify "
        "such stocks and allocate capital accordingly."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or len(history["session_date"].unique()) < self._window:
            return Signal(information_available_at=stamp, weights={})

        means = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .with_columns((pl.col("close") - pl.col("mean")).alias("deviation"))
            .sort("deviation", descending=True)
        )

        top_n_symbols = [row["symbol"] for row in means.to_dicts()[: self._window]]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight
                for symbol in top_n_symbols
                if symbol in view.symbols
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
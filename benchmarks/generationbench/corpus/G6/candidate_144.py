from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy exploits short-term price trends across stocks in the Indian market "
        "using a combination of daily O, H, L, C, and V data. It leverages both mean reversion "
        "and long-term momentum to ensure a balanced approach."
    )

    def __init__(self, window: int = 20, top_n_long: int = 30, top_n_short: int = 30) -> None:
        self._window = window
        self._top_n_long = top_n_long
        self._top_n_short = top_n_short

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        daily_returns = (
            history.with_columns(
                (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date", descending=True)
            .group_by("symbol")
            .agg(pl.col("r").mean().alias("momentum_score"))
        )

        if daily_returns.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = (
            daily_returns.sort("momentum_score", descending=True)
            .select(["symbol"])
            .to_series()
            .to_list()[: self._top_n_long + self._top_n_short]
        )

        long_weights: dict[str, float] = {}
        short_weights: dict[str, float] = {}

        for symbol in sorted_symbols[: self._top_n_long]:
            weight = 0.01 / (2 * self._top_n_long)
            long_weights[symbol] = weight
        for symbol in sorted_symbols[self._top_n_long : self._top_n_long + self._top_n_short]:
            weight = -0.01 / self._top_n_short
            short_weights[symbol] = weight

        weights = {**long_weights, **short_weights}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
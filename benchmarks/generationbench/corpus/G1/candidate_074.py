from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends based on volatility. It identifies symbols with high "
        "volatility over a short term and buys them, assuming they are likely to continue in "
        "the same direction due to higher liquidity and trader interest."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        avg_volatility = (
            history.select(pl.col("adj_close").rolling_std(window=self._window))
            .group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("avg_volatility"))
            .sort("avg_volatility", descending=True)
            .head(5)["avg_volatility"]
            .to_list()
        )

        if len(avg_volatility) < 1:
            return Signal(information_available_at=stamp, weights={})

        thresholded_symbols = [
            symbol
            for symbol in view.symbols
            if float(
                history.select((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"))
                .filter(pl.col("symbol") == symbol)
                .select(pl.col("return").std())
                .item()
            )
            > self._threshold
        ]

        if not thresholded_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(thresholded_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in thresholded_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy exploits the tendency for low-volatility stocks to outperform high-volatility "
        "stocks over time. By constructing a portfolio that favors less volatile stocks and reducing "
        "exposure to more volatile ones, we aim to enhance risk-adjusted returns."
    )

    def __init__(self, window: int = 250, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("close").shift(-1) / pl.col("close") - 1.0).alias("return")
            )
            .drop_nulls()
            .sort("session_date", descending=False)
        )

        # Compute rolling volatility
        volatilities = (
            history.select(pl.exclude("symbol"))
            .groupby(["symbol"])
            .agg(
                (pl.col("return").rolling_std(window_size=self._window).alias("volatility"))
            )
            .sort("session_date", descending=False)
        )

        # Ensure non-null volatility
        volatilities = volatilities.fill_null(value=float("inf"))

        # Rank symbols based on their volatility
        ranked_symbols = (
            volatilities.sort("volatility").select(pl.col("symbol")).head(self._top_n)
        ).to_list()[0]

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_symbols)
        weights = {s: weight for s in ranked_symbols}

        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weights[symbol]
                for symbol in view.symbols
                if symbol in weights
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
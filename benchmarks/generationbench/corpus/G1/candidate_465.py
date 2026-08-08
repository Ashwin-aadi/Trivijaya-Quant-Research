from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility tilting has been shown to generate excess returns by "
        "reducing the impact of market downturns through diversification."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
        )

        # Filter out null values for daily returns
        non_null_history = history.select(
            pl.col("symbol"), pl.col("session_date"), "r"
        ).filter(pl.col("r").is_not_null())

        if non_null_history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate rolling mean and standard deviation of returns
        volatilities = (
            non_null_history.group_by("symbol")
            .agg(
                pl.col("r").mean().alias("mean"),
                (pl.col("r").std()).alias("volatility"),
            )
            .sort("volatility", descending=False)
        )

        top_n_symbols = [row[0] for row in volatilities.select(pl.col("symbol"))[: self._window].to_list()]

        weight = 1.0 / len(top_n_symbols) if top_n_symbols else 0
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
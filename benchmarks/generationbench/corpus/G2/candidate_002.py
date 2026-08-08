from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the idea that assets with higher recent "
        "volatility are more likely to continue their current direction. This strategy aims to "
        "capitalize on trending behavior by identifying symbols that have exhibited strong "
        "trends and allocating capital accordingly."
    )

    def __init__(self, window: int = 20, scaling_factor: float = 1.5) -> None:
        self._window = window
        self._scaling_factor = scaling_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .filter(pl.col("symbol").is_in(view.symbols))
            .sort("session_date", descending=False)
        )

        # Calculate rolling volatility
        volatilities = (
            returns.group_by("symbol")
            .agg(
                (pl.col("r").std().over(window_size=self._window).alias("volatility"))
            )
            .sort("volatility", descending=True)
        )

        # Scale the volatility and find top symbols
        scaled_volatilities = volatilities.with_columns(
            (
                pl.col("volatility") * self._scaling_factor
            ).alias("scaled_volatility")
        )
        top_symbols = [
            row["symbol"]
            for _, row in scaled_volatilities.iter_rows()
            if row["scaled_volatility"] > 0.1  # Threshold for inclusion
        ]

        # Generate the signal weights
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})
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
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "This strategy seeks to capitalize on this phenomenon by overweighting low-volatility "
        "stocks in the portfolio."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the volatility for each stock
        volatilities = (
            history.group_by("symbol")
            .agg(
                (pl.col("adj_close").std().alias("volatility"))
            )
            .select(pl.col("symbol"), pl.col("volatility"))
            .with_columns(
                (1 / pl.col("volatility")).alias("weight")
            )
        )

        # Normalize the weights
        total_weight = volatilities.select(pl.sum("weight").alias("total_weight"))[0, "total_weight"]
        normalized_weights = (
            volatilities.with_columns((pl.col("weight") / total_weight).alias("normalized_weight"))
        ).select(pl.col("symbol"), pl.col("normalized_weight"))

        # Create the signal
        weights = {row["symbol"]: float(row["normalized_weight"]) for row in normalized_weights.to_dict(orient="records")}
        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
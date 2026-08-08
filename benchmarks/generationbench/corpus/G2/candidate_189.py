from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "This is because low-volatility stocks are less risky and often benefit from higher risk-adjusted returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the rolling standard deviation as a measure of volatility
        volatilities = (
            history.group_by("symbol")
            .agg(
                (pl.col("adj_close").std().alias("volatility"))
            )
            .select(["symbol", "volatility"])
        )

        # Sort by volatility and pick the lowest ones
        sorted_volatilities = volatilities.sort("volatility").to_pandas()
        top_symbols = sorted_volatilities["symbol"].head(5).tolist()

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Assign equal weight to each selected stock
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
from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "High-volatility stocks tend to exhibit mean reversion or trend continuation. By "
        "identifying the most volatile stocks and allocating capital accordingly, we can "
        "capture returns from both phenomena."
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
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
        )
        # Calculate volatility over the window period
        volatilities = (
            history.group_by("symbol")
            .agg(pl.col("returns").std().alias("volatility"))
            .sort("volatility", descending=True)
        )

        if volatilities.height < 1:
            return Signal(information_available_at=stamp, weights={})

        top_symbol = volatilities.select("symbol")[0]
        weight = 1.0
        return Signal(
            information_available_at=stamp,
            weights={top_symbol: weight},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
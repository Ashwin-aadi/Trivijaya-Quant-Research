from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are often indicative of strong buying or selling "
        "pressure. By identifying such moves early, we can capitalize on potentially significant "
        "price trends."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history_with_returns = (
            history
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date")
        )

        # Filter out rows with no return data
        non_zero_returns = history_with_returns.filter(pl.col("return") != 0.0)

        if non_zero_returns.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Group by symbol and calculate the sum of volumes on days with returns
        volume_sum = (
            non_zero_returns
            .group_by("symbol")
            .agg(
                pl.col("volume").sum().alias("total_volume"),
                (pl.col("return") * pl.col("volume")).sum().alias("volume_return"),
            )
        )

        # Calculate the average return per symbol
        avg_return = (
            volume_sum
            .with_columns(
                (pl.col("volume_return") / pl.col("total_volume")).alias("avg_return")
            )
            .select(["symbol", "avg_return"])
        )

        # Identify symbols with the highest average returns
        top_symbols = avg_return.sort("avg_return", descending=True).head(5)

        if top_symbols.is_empty():
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={row["symbol"]: weight for row in top_symbols.rows()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).item()
    assert isinstance(newest, date)
    return newest
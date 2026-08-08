from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthStrategy(Strategy):
    rationale = (
        "Stocks with higher returns compared to the broader market tend to outperform over "
        "the long term. This strategy selects top-performing stocks based on their relative strength."
    )

    def __init__(self, window: int = 20, num_assets: int = 5) -> None:
        self._window = window
        self._num_assets = num_assets

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window or len(history["symbol"].unique()) < 2:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date", descending=True)
            .drop_nulls(subset=["symbol"])
        )

        # Calculate mean returns for the entire universe
        mean_returns = history.groupby("symbol").agg(
            (pl.col("r").mean().alias("mean_r"))
        ).select("symbol", "mean_r")

        # Join to get symbol-wise mean returns and daily returns
        merged_history = (
            history.join(mean_returns, on="symbol")
            .with_columns((pl.col("r") / pl.col("mean_r")).alias("relative_strength"))
            .sort("session_date", descending=True)
        )

        # Get the top N assets based on relative strength
        top_assets = merged_history.top_k(self._num_assets, by="relative_strength")

        if not top_assets.height:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / self._num_assets
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_assets["symbol"].to_list()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select("session_date").max().to_list()[0]
    assert isinstance(newest, date)
    return newest
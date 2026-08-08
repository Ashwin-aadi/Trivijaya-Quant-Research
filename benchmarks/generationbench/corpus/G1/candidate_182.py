from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the idea that assets with higher "
        "volatility should be traded more aggressively. By scaling trends by their volatility, "
        "we aim to capture strong movements while mitigating risk."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns and rolling volatility
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .group_by("symbol")
            .agg(
                (pl.col("return").mean().alias("mean_return")),
                (pl.col("return").std().alias("volatility")),
            )
        )

        # Find symbols with high mean returns and low volatility
        picks = (
            history.filter(
                (pl.col("mean_return") > 0) & (pl.col("volatility") < self._threshold * pl.col("mean_return"))
            )
            .sort("mean_return", descending=True)
            .select("symbol")
            .head(self._window)
        )

        if picks.height == 0:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / picks.height
        return Signal(
            information_available_at=stamp,
            weights={p: weight for p in picks.to_list()["symbol"]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
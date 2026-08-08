from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy aims to follow trends by using a volatility-scaled moving average. "
        "High volatility periods suggest increased uncertainty, and the trend may be less reliable; "
        "conversely, low volatility suggests a more reliable trend can be followed."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.groupby("symbol").agg(
            (pl.col("adj_close").mean().alias("mean_close"))
        )
        std_close = history.groupby("symbol").agg(
            (pl.col("adj_close").std().alias("std_close"))
        )

        # Combine the two DataFrames to have both mean and std in one
        combined_history = mean_close.join(std_close, on="symbol")
        trend_scores = (
            combined_history.with_columns(
                (pl.col("adj_close") - pl.col("mean_close")) / (pl.col("std_close") * self._threshold).alias("trend_score")
            )
            .select("symbol", "trend_score")
            .sort("trend_score", descending=True)
            .to_dict(False)
        )

        if not trend_scores:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [score["symbol"] for score in trend_scores[:5]]
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
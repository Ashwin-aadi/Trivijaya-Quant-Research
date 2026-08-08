from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong momentum. "
        "By identifying symbols that show a significant price move with high volume, we can capitalize on trending behavior."
    )

    def __init__(self, window: int = 30, threshold: float = 0.02) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            history.lazy()
            .with_columns((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r"))
            .group_by("symbol")
            .agg(
                (pl.col("r").mean().alias("avg_return")),
                (pl.col("volume").sum().alias("total_volume")),
                (pl.col("r").std().alias("std_return")),
            )
            .collect()
        )

        # Filter to find symbols with significant returns
        significant_returns = (
            returns.filter(
                ((pl.col("avg_return") > self._threshold) & (pl.col("total_volume") >= 10_000))
                | ((pl.col("avg_return") < -self._threshold) & (pl.col("total_volume") >= 10_000))
            ).sort("symbol").select(pl.col("symbol"))
        )

        if significant_returns.is_empty():
            return Signal(information_available_at=stamp, weights={})

        top_symbols = significant_returns["symbol"].to_list()[:5]
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
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy aims to exploit the tendency for stocks with higher liquidity to outperform "
        "those with lower liquidity in the Indian market. By equal weighting stocks that pass a "
        "liquidity screen (e.g., trading volume and turnover ratio thresholds), we seek to capture "
        "the positive returns associated with liquidity while maintaining diversification across "
        "stocks."
    )

    def __init__(self, window: int = 30, volume_threshold: float = 5_000_000) -> None:
        self._window = window
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Compute liquidity metrics
        history = (
            history.with_columns(
                (pl.col("volume") / pl.col("adj_close").mean().over("symbol")).alias("turnover_ratio")
            )
            .sort("session_date", descending=True)
            .group_by("symbol")
            .agg(
                pl.col("volume").sum().alias("total_volume"),
                (pl.col("volume") / pl.col("adj_close").mean().over("symbol")).alias("turnover_ratio"),
            )
        )

        # Filter by volume threshold
        history = history.filter(pl.col("total_volume") > self._volume_threshold)

        # Equal weighting of the remaining stocks
        symbols = [s for s in view.symbols if s in history.columns]
        weights = {s: 1.0 / len(symbols) for s in symbols}

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
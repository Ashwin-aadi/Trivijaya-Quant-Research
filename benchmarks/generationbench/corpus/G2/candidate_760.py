from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Price reversion strategies exploit mean reversion in asset prices. "
        "If a stock's price consistently returns to its trailing mean, it can lead to profitable trades."
    )

    def __init__(self, window: int = 20, z_score_threshold: float = 1.5) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        means = history.groupby("symbol").agg(
            (pl.col("close").mean().alias("trailing_mean"))
        )
        latest_closes = view.closes()

        # Join to get trailing mean for each symbol in the latest close
        merged = (
            means.join(latest_closes, on="symbol", how="inner")
            .with_columns(
                (pl.col("close") - pl.col("trailing_mean")).alias("deviation"),
                ((pl.col("close") - pl.col("trailing_mean"))
                 / pl.col("close").std().over("symbol")).alias("z_score"),
            )
        )

        # Filter for symbols with z-score above the threshold
        picks = (
            merged.filter(
                (pl.col("z_score").abs() > self._z_score_threshold)
                & (pl.col("deviation") < 0)  # Sell if price is below trailing mean and z-score is high
            )
            .select("symbol")
            .to_series()
        ).to_list()

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
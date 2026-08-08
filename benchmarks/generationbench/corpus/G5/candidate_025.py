from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion strategies exploit the tendency of financial assets to return to their "
        "mean price level over time. Short-term deviations from this mean are expected to revert, "
        "offering trading opportunities."
    )

    def __init__(self, window: int = 5, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        means = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .with_column(
                (pl.col("adj_close") - pl.col("mean")).abs()
                .rank(method="ordinal", descending=True)
                .alias("deviation_rank")
            )
        )

        # Filter to get symbols with deviations within a threshold
        filtered_means = (
            means.filter(pl.col("deviation_rank").lt(self._threshold))
            .select(["symbol"])
            .to_dict(as_series=False)["symbol"]
        )

        if not filtered_means:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_means)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in filtered_means},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
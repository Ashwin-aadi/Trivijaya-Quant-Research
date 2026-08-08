from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion strategies seek to identify stocks that have deviated significantly "
        "from their historical mean price and are likely to revert to it. This strategy uses a "
        "short lookback period to detect such deviations."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean()).alias("mean"))
            .with_column(
                (pl.col("adj_close") - pl.col("mean")).abs().alias("deviation")
            )
            .select(pl.all(), (pl.col("deviation") / self._threshold).alias("scaled_deviation"))
        )

        # Filter symbols with high deviation
        top_deviations = mean_close.sort("scaled_deviation", descending=True).head(5)

        picks: list[str] = []
        for symbol in view.symbols:
            if (
                float(top_deviations[top_deviations["symbol"] == symbol]["scaled_deviation"])
                > 1.0
            ):
                picks.append(symbol)

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
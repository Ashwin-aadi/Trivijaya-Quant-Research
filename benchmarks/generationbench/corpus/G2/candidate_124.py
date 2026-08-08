from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingAverage(Strategy):
    rationale = (
        "Price reverts to its mean over time. By identifying stocks that have deviated significantly "
        "from their trailing average price level, we can predict a return towards the mean."
    )

    def __init__(self, window: int = 20, deviation_threshold: float = 1.5) -> None:
        self._window = window
        self._deviation_threshold = deviation_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate trailing average and deviations
        avg_close = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean()).alias("trailing_avg"))
            .join(history, on="symbol", how="left")
            .with_column(
                (pl.col("adj_close") / pl.col("trailing_avg") - 1.0).alias("deviation")
            )
        )

        # Filter symbols based on deviation threshold
        avg_close = avg_close.filter((pl.col("deviation").abs() >= self._deviation_threshold))
        picks: list[str] = [row["symbol"] for row in avg_close.to_dicts()]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price levels revert to a mean. By tracking the trailing 20-day average, we can "
        "identify stocks that have deviated significantly from their recent mean price and "
        "are likely to return towards it."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate the trailing average
        avg_close = (
            history.groupby("symbol")
                   .agg((pl.col("adj_close").mean()).alias("avg_close"))
                   .join(closes, on="symbol")
        )
        
        # Compute deviations from the mean
        avg_close = avg_close.with_columns(
            (pl.col("adj_close") / pl.col("avg_close") - 1.0).alias("deviation")
        )

        # Identify symbols with significant deviation
        picks: list[str] = [row["symbol"] for row in avg_close.filter(
            (pl.col("deviation") > 2) | (pl.col("deviation") < -2)
        ).to_dict(False)]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
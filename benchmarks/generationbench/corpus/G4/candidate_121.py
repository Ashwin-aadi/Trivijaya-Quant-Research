from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "This strategy exploits short-horizon mean reversion by identifying stocks whose "
        "closing prices deviate significantly from their 10-day simple moving average (SMA). "
        "Buy signals are generated when prices fall below the SMA, and sell signals when they"
        " rise above it. The aim is to profit from the return to normal pricing levels."
    )

    def __init__(self, window: int = 10, threshold: float = 0.02) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the 10-day SMA
        sma_10d = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean()).alias("sma_10d"))
            .with_columns(
                (pl.col("close") / pl.col("sma_10d") - 1.0).alias("deviation"),
                ((pl.col("close") - pl.col("sma_10d")) / pl.col("sma_10d")).abs().alias("abs_deviation")
            )
        )

        # Filter for significant deviations
        filtered = sma_10d.filter(
            (pl.col("deviation").lt(-self._threshold)) | 
            (pl.col("deviation").gt(self._threshold))
        ).select(["symbol", "abs_deviation"])

        if filtered.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Rank symbols by absolute deviation
        ranked = filtered.with_column(
            pl.col("abs_deviation").rank(method="dense", descending=True).alias("rank")
        )

        top_symbols = ranked.sort("rank").select(["symbol"])[0].to_list()

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
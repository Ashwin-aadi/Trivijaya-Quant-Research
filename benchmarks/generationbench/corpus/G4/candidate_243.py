from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy focuses on exploiting the liquidity premium by selecting a diversified "
        "portfolio of stocks screened for high trading volume. High liquidity is associated with "
        "better execution quality and lower market impact costs, leading to higher observed returns."
    )

    def __init__(self, window: int = 30, top_n: int = 100) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate average volume for each stock over the lookback period
        avg_volume = (
            history.group_by("symbol")
                   .agg((pl.col("volume") / self._window).alias("avg_volume"))
                   .sort("avg_volume", descending=True)
        )

        if avg_volume.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [str(sym) for sym in avg_volume.select(pl.col("symbol")).head(self._top_n)["symbol"]]
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
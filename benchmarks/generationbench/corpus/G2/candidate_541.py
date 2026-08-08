from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion occurs when security prices return to the long-term average price. "
        "In this context, if a stock's price has significantly deviated from its mean over a 20-day period, "
        "it is expected to revert to that mean in the near future."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.select(
            pl.col("symbol").alias("symbol"),
            (pl.col("adj_close").mean().over("symbol")).alias("mean"),
        ).group_by("symbol")

        deviations = history.join(mean_close, on="symbol", how="left")
        deviations = (
            deviations.with_columns((pl.col("adj_close") - pl.col("mean")).abs().alias("deviation"))
            .sort("deviation", descending=True)
            .select(["session_date", "symbol"])
        )

        if len(deviations) < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = deviations.head(self._window)["symbol"].to_list()
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
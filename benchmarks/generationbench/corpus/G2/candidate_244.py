from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion occurs when prices return to their mean after a deviation. "
        "In an efficient market, deviations from the mean are temporary and revert back. "
        "By identifying stocks that have significantly deviated from their historical price range, "
        "we can exploit this reversion effect."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the mean and standard deviation of the last `window` days
        mean_price = history.select(
            pl.col("adj_close").mean().alias("mean")
        ).get_column("mean")[0]
        std_dev = history.select(
            (pl.col("adj_close") - pl.col("mean")).std().alias("std")
        ).get_column("std")[0]

        # Identify symbols that are outside the mean ± 2 * std range
        filtered_history = (
            history.with_columns((pl.col("adj_close") - mean_price).abs().alias("deviation"))
            .with_columns((pl.col("deviation") / std_dev).alias("z_score"))
            .filter(pl.col("z_score").gt(2.0) | pl.col("z_score").lt(-2.0))
        )

        if filtered_history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in filtered_history.columns]
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
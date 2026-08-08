from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency of stocks that have performed "
        "well in the recent past to continue outperforming their peers. This strategy looks "
        "for the top-performing stocks over a certain period and allocates capital towards them."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < 2 * self._window:
            return Signal(information_available_at=stamp, weights={})

        # Compute daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date")
            .select(["symbol", "session_date", "return"])
        )

        # Calculate cumulative returns
        history = (
            history.with_column(
                (pl.col("return") + 1.0).cumprod().alias("cumulative_return")
            )
            .group_by("symbol")
            .agg(pl.col("cumulative_return").max().alias("max_cumulative_return"))
            .sort("max_cumulative_return", descending=True)
        )

        # Select the top N symbols
        history = history.head(self._top_n)

        if history.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(history)
        return Signal(
            information_available_at=stamp,
            weights={row["symbol"]: weight for row in history.iter_rows()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
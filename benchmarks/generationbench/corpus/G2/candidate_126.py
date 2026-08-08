from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "The relative strength (RS) strategy exploits the idea that stocks "
        "with strong past performance are more likely to continue outperforming. "
        "This is based on the assumption that successful firms tend to maintain their "
        "growth trajectory over time."
    )

    def __init__(self, lookback: int = 60, top_n: int = 10) -> None:
        self._lookback = lookback
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.height < self._lookback or len(history.columns) <= 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
            )
            .sort("session_date", descending=False)
            .drop_nulls()
        )

        # Compute the average returns for each symbol
        avg_returns = (
            history.groupby("symbol")
            .agg(
                pl.col("returns").mean().alias("avg_return"),
                (pl.col("returns") >= 0).sum().alias("win_count"),
            )
            .sort("avg_return", descending=True)
            .head(self._top_n + 1)  # Take top n and an extra to handle ties
        )

        if avg_returns.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        winners: list[str] = [row.symbol for row in avg_returns.rows()][:self._top_n]

        weight = 1.0 / len(winners)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in winners},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
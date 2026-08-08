from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects the top-performing stocks based on their relative strength "
        "against the broad market index. The idea is to focus on stocks that have outperformed "
        "the NIFTY 100 over a certain period."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date")
            .drop_nulls()
        )

        # Group by symbol and calculate the mean return
        means = (
            history.group_by("symbol")
            .agg((pl.col("return").mean().alias("avg_return")))
            .collect()
        )

        # Filter out symbols with missing returns or those that are not in view.symbols
        valid_symbols = [s for s in view.symbols if s in means["symbol"].to_list()]
        means = means.filter(pl.col("symbol").is_in(valid_symbols)).sort("avg_return", descending=True)

        top_n = min(self._window, len(means))
        picks: list[str] = [str(row[0]) for row in means.head(top_n).rows()]

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
from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy identifies stocks that have outperformed their peers over a recent "
        "period and invests in them. It relies on the persistence of stock performance, "
        "where high past returns often continue into the future due to factors like firm-specific "
        "events or market sentiment effects."
    )

    def __init__(self, window: int = 250) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("daily_return")
            )
            .sort("session_date", descending=False)
            .drop_nulls()
        )

        # Calculate cumulative returns over the lookback period
        history = (
            history.with_columns(
                (pl.col("daily_return").sum()).alias("cumulative_return")
            )
            .sort("symbol", "session_date", descending=True)
            .group_by("symbol")
            .agg(pl.col("cumulative_return").last().alias("cumulative_return"))
        )

        # Rank stocks by cumulative return
        ranking = history.sort("cumulative_return", descending=True).to_numpy()

        top_n = 20
        bottom_n = 15

        long_symbols = [ranking[i][0] for i in range(top_n)]
        short_symbols = [ranking[i + (top_n + bottom_n)][0] for i in range(bottom_n)]

        if not long_symbols or not short_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_long = 1.0 / len(long_symbols)
        weight_short = -1.0 / len(short_symbols)

        weights = {s: weight_long for s in long_symbols}
        for s in short_symbols:
            if s not in weights:
                weights[s] = weight_short

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
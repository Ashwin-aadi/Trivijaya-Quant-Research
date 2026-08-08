from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Momentum3m(Strategy):
    rationale = (
        "This strategy capitalizes on the phenomenon that stocks with strong recent "
        "performance are likely to continue outperforming. By ranking Indian-listed "
        "companies based on their cumulative returns over a 3-month period and selecting "
        "the top-performing stocks, we aim to benefit from momentum persistence."
    )

    def __init__(self, window: int = 90, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .group_by("symbol")
            .agg(pl.sum("return").alias("cumulative_return"))
        )

        # Filter out symbols with no return data
        history = history.filter(~pl.col("cumulative_return").is_nan())

        # Rank symbols by cumulative return
        ranks = (
            history.sort("cumulative_return", descending=True)
            .group_by("symbol")
            .agg(pl.col("cumulative_return").rank(method="dense", descending=True).alias("rank"))
        )

        top_symbols = ranks.filter(pl.col("rank") <= self._top_n)["symbol"].to_list()

        # Equal weighting for selected symbols
        weights = {s: 1.0 / len(top_symbols) for s in top_symbols}

        return Signal(
            information_available_at=stamp,
            weights={s: weights[s] if s in weights else 0.0 for s in view.symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
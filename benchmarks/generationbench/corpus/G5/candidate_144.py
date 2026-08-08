from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum identifies stocks that have outperformed the market "
        "recently and allocates capital to them, leveraging the idea that past winners are likely"
        "to continue performing well."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("r")
            )
            .group_by("symbol")
            .agg((pl.col("r").mean().alias("avg_return")))
            .sort(by="avg_return", descending=True)
        )

        if history.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = [row["symbol"] for row in history.to_dicts()[: self._top_n]]
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
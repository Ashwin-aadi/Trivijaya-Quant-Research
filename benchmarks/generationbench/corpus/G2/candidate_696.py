from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum seeks to identify stocks that have outperformed their peers "
        "over a recent period. This strategy benefits from the tendency of strong performers to "
        "continue performing well in the future."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns for each stock
        history = (
            history.with_columns(
                (pl.col("close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("average_return"))
        )

        # Select top performers
        sorted_history = history.sort("average_return", descending=True)
        symbols = [row["symbol"] for row in sorted_history.to_dicts()][:5]

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

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
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum involves selecting assets that have outperformed "
        "their peers over a recent period. This strategy leverages the notion that stocks "
        "that have performed well recently are likely to continue performing well."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns
        returns = (
            history.lazy()
            .with_columns(
                (pl.col("close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
            .collect()
        )

        # Sort by average return and pick top performers
        sorted_returns = returns.sort("avg_return", descending=True)
        top_symbols = [row["symbol"] for row in sorted_returns.to_dicts()][:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
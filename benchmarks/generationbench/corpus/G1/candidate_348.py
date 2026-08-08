from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy selects the top-performing stocks based on recent returns and allocates "
        "capital to them. The rationale is that strong relative performance over a short period "
        "indicates potential for continued outperformance."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            history
            .sort("session_date")
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
            )
        )

        # Group by symbol and calculate mean return over the window
        means = (
            returns
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
        )

        # Sort by average return in descending order
        ranked = means.sort("avg_return", descending=True)

        if ranked.height < 5:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in ranked.head(5).to_dicts()]
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
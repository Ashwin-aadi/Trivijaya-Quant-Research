from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "relative to their peers in recent periods to continue outperforming. This strategy "
        "identifies top performers and allocates capital accordingly."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns for each symbol
        returns = (
            closes
            .lazy()
            .with_column(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0)
                .alias("return")
            )
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
            .sort("avg_return", descending=True)
            .collect()
        )

        top_symbols = [row["symbol"] for row in returns.rows()[:self._top_n]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
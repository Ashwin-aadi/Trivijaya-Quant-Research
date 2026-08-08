from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the idea that assets with high relative strength "
        "tend to continue outperforming in the near future. This is based on the notion that "
        "markets tend to reward strong performers."
    )

    def __init__(self, lookback_days: int = 60, top_n: int = 10) -> None:
        self._lookback_days = lookback_days
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.height < self._lookback_days:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._lookback_days)

        # Compute the returns for each symbol
        returns = (
            closes.drop_nulls()
            .select(
                pl.col("session_date"),
                (pl.col(pl.Series(closes.columns[1:])) / pl.col(pl.Series(closes.columns[:-1])) - 1.0)
                .alias("returns")
            )
            .with_column((pl.col("returns").rank(method="ordinal", descending=True)).alias("rank"))
        )

        # Filter out symbols with missing data
        filtered_returns = returns.filter(
            pl.all().not_null()
        ).group_by("session_date").agg(pl.col("symbol"), pl.col("rank"))

        top_symbols = (
            filtered_returns.sort("rank")
            .filter(pl.col("rank") <= self._top_n)
            .select("symbol")
            .to_series()
            .to_list()
        )

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

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
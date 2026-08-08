from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum by identifying stocks that have "
        "outperformed in recent periods and expecting them to continue outperforming. It is "
        "based on the assumption that past performance is indicative of future returns due to "
        "investor sentiment and market efficiency dynamics."
    )

    def __init__(self, lookback_days: int = 90, top_n: int = 25) -> None:
        self._lookback_days = lookback_days
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Compute log returns
        history = (
            history
            .with_column(
                (pl.col("close").shift(-self._lookback_days) / pl.col("close") - 1.0).alias("log_return")
            )
            .sort("session_date", descending=False)
            .drop_nulls()
        )

        # Calculate cumulative log returns over the lookback period
        history = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("log_return").sum().alias("cumulative_return"))
            )
        )

        # Sort by cumulative return and select top N stocks
        ranked_symbols = history.sort("cumulative_return", descending=True)["symbol"].to_list()[: self._top_n]

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
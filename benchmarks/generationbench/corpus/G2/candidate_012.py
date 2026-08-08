from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have outperformed "
        "their peers in recent periods to continue outperforming. This is based on the idea "
        "that market inefficiencies can lead to persistent performance differences between "
        "stocks."
    )

    def __init__(self, window: int = 60, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate simple returns for the latest window
        history = (
            history.with_columns(
                (pl.col("close") / pl.col("close").shift(self._window) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .head(self._lookback)
        )

        # Rank symbols by return
        ranked = (
            history.group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
            .with_columns(
                pl.col("avg_return").rank(method="ordinal", descending=True).alias("rank")
            )
        )

        # Select top N performers
        top_n_symbols = ranked.sort("rank").select(["symbol"]).to_series().to_list()[: self._lookback]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
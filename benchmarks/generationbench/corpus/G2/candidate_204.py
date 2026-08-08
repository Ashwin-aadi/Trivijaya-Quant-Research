from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum suggests that assets with strong historical returns "
        "outperform those with weaker returns. This strategy exploits the idea by "
        "buying assets in the top decile of past performance."
    )

    def __init__(self, lookback_period: int = 60, top_decile_size: float = 0.1) -> None:
        self._lookback_period = lookback_period
        self._top_decile_size = top_decile_size

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if len(symbols) < self._top_decile_size * 10:
            return Signal(information_available_at=stamp, weights={})

        closes = history.select(pl.col("adj_close").exclude("session_date"))
        returns = (
            closes.shift(-self._lookback_period)
            / closes
            - 1.0
        ).sort("symbol", descending=True)

        rank = (
            returns.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .group_by("symbol")
            .agg((pl.col("return").rank(method="ordinal", descending=True)).alias("rank"))
        )

        threshold = rank.height * self._top_decile_size
        top_symbols = [s for s, r in zip(rank["symbol"], rank["rank"].to_list()) if r < threshold]

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
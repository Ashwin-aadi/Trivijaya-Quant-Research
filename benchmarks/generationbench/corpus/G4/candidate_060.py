from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Momentum30d(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum by selecting stocks with the highest "
        "returns over a 30-day look-back period. Higher returns indicate strong past performance, "
        "which often persists in the short term due to investor behavior and market dynamics."
    )

    def __init__(self, window: int = 30, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or len(view.symbols) == 0:
            return Signal(information_available_at=stamp, weights={})

        returns = (
            closes.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return"),
            )
            .collect()["return"]
            .to_list()
        )

        if len(returns) < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        ranked_returns = sorted(zip(view.symbols, returns), key=lambda x: -x[1])
        top_symbols = [symbol for symbol, _ in ranked_returns[:self._top_n]]

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
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum focuses on selecting assets with the strongest recent "
        "price appreciation. By leveraging logarithmic returns over a 30-day window, we can "
        "identify stocks that have outperformed their peers, providing opportunities for "
        "positive alpha."
    )

    def __init__(self, window: int = 30, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        log_returns: pl.DataFrame = (
            history.lazy()
            .select(pl.col("symbol"), (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("log_return"))
            .group_by("symbol")
            .agg((pl.col("log_return").mean().alias("avg_log_return")))
            .collect()
        )

        top_symbols = log_returns.sort("avg_log_return", descending=True)["symbol"].to_list()[: self._top_n]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 0.1
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
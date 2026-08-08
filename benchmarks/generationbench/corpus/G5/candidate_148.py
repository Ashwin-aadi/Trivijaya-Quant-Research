from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price reverts towards the mean over time. By identifying symbols that have moved "
        "farthest from their trailing average and then trading in the direction of the mean, "
        "we can exploit this tendency for profit."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or not closes.columns:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            closes.select(pl.col("adj_close").mean())
            .to_series()
            .item()
        )
        deviations = [
            (float(closes[symbol].max()) - mean_close) / mean_close
            for symbol in view.symbols
            if symbol in closes.columns and not closes[symbol].is_empty()
        ]

        sorted_symbols = [s for _, s in sorted(zip(deviations, view.symbols), reverse=True)]
        top_n = sorted_symbols[: self._top_n]

        if not top_n:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
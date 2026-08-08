from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class PriceLevelReversion(Strategy):
    rationale = (
        "Price reversion strategies aim to identify securities that have deviated significantly "
        "from their historical price levels and expect them to revert to those levels. This strategy "
        "focuses on identifying stocks with the highest absolute deviation from their 20-day mean, "
        "hoping for a return towards this mean."
    )

    def __init__(self, window: int = 20, top_n: int = 5, threshold: float = 2.0) -> None:
        self._window = window
        self._top_n = top_n
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or "adj_close" not in closes.columns:
            return Signal(information_available_at=stamp, weights={})

        mean_price = closes["adj_close"].mean().to_list()[0]
        deviations = [
            (float(c) / mean_price - 1.0) * 100 for c in closes["adj_close"].drop_nulls().to_list()
        ]
        if len(deviations) < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [
            symbol
            for _, (symbol, dev) in sorted(
                zip(view.symbols, deviations), key=lambda x: abs(x[1]), reverse=True
            )
            if abs(dev) >= self._threshold
        ][: self._top_n]
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
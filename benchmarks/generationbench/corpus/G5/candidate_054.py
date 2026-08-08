from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks with the highest relative strength compared to their own historical "
        "performance can provide an edge in equity selection. Stocks that consistently outperform "
        "their peers are likely to continue this trend and offer better returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        strength_scores = {}
        for symbol in symbols:
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            last_close = values[-1]
            average_close = sum(values) / len(values)
            relative_strength = (last_close - average_close) / (
                max(values) - min(values)
            )
            strength_scores[symbol] = relative_strength

        top_symbols = sorted(strength_scores.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
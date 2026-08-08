from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Identify stocks that are outperforming the broader market by selecting "
        "the top N stocks with the highest return relative to the index average."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or len(view.symbols) < 2:
            return Signal(information_available_at=stamp, weights={})

        avg_return = (
            (closes["close"] / closes["close"].shift(self._window - 1) - 1.0).mean()
        )
        symbol_returns = [
            float(c / closes[c].shift(self._window - 1) - 1.0)
            for c in closes.columns[1:]
        ]
        rank_order = sorted(
            zip(view.symbols, symbol_returns), key=lambda x: x[1], reverse=True
        )[: self._top_n]

        picks = [symb for symb, _ in rank_order]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
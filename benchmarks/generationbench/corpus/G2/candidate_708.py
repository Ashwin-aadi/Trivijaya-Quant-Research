from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum suggests that assets with the highest returns over a "
        "recent period are likely to continue outperforming. This strategy exploits the idea "
        "that past performance can predict future performance."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        performance = (
            closes[[symbol for symbol in view.symbols]]
            .select(
                [
                    pl.col(symbol).last() / pl.col(symbol).shift(self._window) - 1.0
                    for symbol in view.symbols
                ]
            )
            .row(0)
            .to_list()
        )

        top_symbols = [symbol for _, symbol in sorted(zip(performance, view.symbols), reverse=True)[: self._top_n]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={symbol: weight for symbol in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
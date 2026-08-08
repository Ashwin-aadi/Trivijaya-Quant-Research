from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression is a sign of reduced volatility and increased market consolidation. "
        "This can be an opportunity to enter positions in stocks that are not showing significant "
        "price action but have been consolidating."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        high_min = (history[s] / history[s].shift(1) - 1.0).min().to_list()[-1]
        low_max = (-history[s] / history[s].shift(1) + 1.0).max().to_list()[-1]

        compressed_symbols: list[str] = []
        for symbol in symbols:
            if high_min >= 0.05 and low_max <= -0.05:
                compressed_symbols.append(symbol)

        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in compressed_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
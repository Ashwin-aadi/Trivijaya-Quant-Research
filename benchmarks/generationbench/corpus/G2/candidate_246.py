from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Securities with a higher relative strength compared to their peers tend to outperform "
        "over the long run. This is based on the idea that strong stocks often maintain or "
        "increase their relative advantage over time."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_strengths = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            symbol_closes = [float(v) for v in closes[symbol].to_list()]
            avg_close = sum(symbol_closes) / len(symbol_closes)
            strength = sum([1 if c > avg_close else 0 for c in symbol_closes]) / self._window
            symbol_strengths[symbol] = strength

        sorted_strengths = {k: v for k, v in sorted(symbol_strengths.items(), key=lambda item: item[1], reverse=True)}
        top_symbols = list(sorted_strengths.keys())[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "relative to the market in the recent past to continue performing well. This strategy "
        "selects the top-performing stocks based on their relative strength over a 20-day period."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window or not all(symbol in closes.columns for symbol in view.symbols):
            return Signal(information_available_at=stamp, weights={})

        symbol_values: dict[str, float] = {}
        for symbol in view.symbols:
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            recent_close = float(closes[stamp, symbol])
            returns = [(recent_close / value - 1.0) * 100.0 for value in values[:-1]]
            momentum_score = max(returns)
            symbol_values[symbol] = momentum_score

        top_symbols = sorted(symbol_values.items(), key=lambda x: x[1], reverse=True)[: self._top_n]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, _ in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
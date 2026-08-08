from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength tend to outperform over time. "
        "By identifying the top performers relative to their peers, we can construct a portfolio of strong stocks."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate relative strength as the ratio of current close to the minimum price in the window
        rel_strengths: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            min_price = min(values)
            current_price = values[-1]
            relative_strength = (current_price / min_price) - 1.0
            rel_strengths[symbol] = relative_strength

        top_symbols: list[str] = []
        for symbol, strength in sorted(rel_strengths.items(), key=lambda x: x[1], reverse=True):
            if strength > 0:
                top_symbols.append(symbol)
            if len(top_symbols) >= self._top_n:
                break

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalBreakout(Strategy):
    rationale = (
        "Certain market conditions exhibit periodic patterns due to seasonal effects. "
        "Identifying and exploiting these patterns can provide trading opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(recent_closes) < self._window:
                continue

            # Check if the latest close is higher than the max of the last 20 days
            if recent_closes[-1] >= max(recent_closes):
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
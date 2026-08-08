from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Identifying a breakout from a 7-day high can indicate strong upward momentum. "
        "This strategy looks for continuation of the breakout by tracking daily highs and "
        "generating signals based on significant percentage increases in price."
    )

    def __init__(self, window: int = 7, threshold: float = 1.05) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            highs = [float(v) for v in history[symbol].sort("session_date").select("high").to_list()[0]]
            if len(highs) < self._window + 1:
                continue
            breakout_high = max(highs[:-1])
            current_high = highs[-1]
            if current_high / breakout_high >= self._threshold:
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
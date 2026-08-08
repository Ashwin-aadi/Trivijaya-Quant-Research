from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "A breakout followed by a confirmation day where the price closes above the high of "
        "the previous trading range suggests continued momentum and strength in the asset. "
        "This strategy aims to capitalize on such confirmatory signals."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].to_list()]
            high_value = max(values[-self._window:])
            last_close = float(history[symbol][-1])
            if last_close > high_value:
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        continuation_symbols: list[str] = []
        for symbol in breakout_symbols:
            history_symbol = view.history(lookback=self._window)[symbol]
            confirmation_high = float(history_symbol[-1])
            previous_high = float(history_symbol[len(history_symbol) - 2])
            if confirmation_high > previous_high and confirmation_high > (previous_high + high_value) / 2:
                continuation_symbols.append(symbol)

        weight = 1.0 / len(continuation_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in continuation_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
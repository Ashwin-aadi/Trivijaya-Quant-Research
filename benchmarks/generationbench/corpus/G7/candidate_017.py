from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation20d(Strategy):
    rationale = (
        "A breakout from a 20-day high or low often indicates a continuation of the trend. "
        "By identifying such breakouts, we can capture potential momentum in the market."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            data = [float(v) for v in history[symbol].to_list()]
            high20 = max(data[-self._window:])
            low20 = min(data[-self._window:])
            latest_close = float(history[history["session_date"] == stamp]["adj_close"][symbol])
            if latest_close > high20:
                breakout_symbols.append(symbol)
            elif latest_close < low20:
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
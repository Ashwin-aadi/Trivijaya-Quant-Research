from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in the direction of initial momentum. By identifying "
        "breakout symbols and analyzing their subsequent behavior, we can capitalize on "
        "the continuation pattern."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns or len(history[symbol].to_list()) < self._window + 1:
                continue
            recent_closes = [float(v) for v in history[symbol][-self._window:].to_list()]
            if recent_closes[-1] >= max(recent_closes):
                breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[: self._top_n]
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        continuation_weights: dict[str, float] = {}
        for symbol in breakout_symbols:
            recent_history = history[symbol][-self._window - 1 : -1].to_list()
            if len(recent_history) < self._window:
                continue
            prior_close = float(recent_history[0])
            subsequent_closes = [float(v) for v in recent_history[1:]]
            trend = sum(subsequent_closes[i] > prior_close for i in range(len(subsequent_closes))) - \
                    sum(subsequent_closes[i] < prior_close for i in range(len(subsequent_closes)))
            if trend > 0:
                weight = 1.0 / len(breakout_symbols)
                continuation_weights[symbol] = weight

        return Signal(
            information_available_at=stamp, weights=continuation_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts that continue to be validated by subsequent prices suggest strong market "
        "sentiment and can lead to persistent trends. Identifying such breakouts allows for "
        "capitalizing on the momentum."
    )

    def __init__(self, window: int = 20, threshold: float = 1.05) -> None:
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
            last_close = float(view.latest_close()[symbol])
            prev_high = float(history[symbol]["high"].max())
            if last_close >= (prev_high * self._threshold):
                breakout_symbols.append(symbol)

        breakout_symbols = list(set(breakout_symbols))  # Remove duplicates

        continuation_symbols: list[str] = []
        for symbol in breakout_symbols:
            if symbol not in history.columns:
                continue
            recent_closes = [float(v) for v in history[symbol]["close"].to_list()[-self._window:]]
            is_continuing = all(
                close >= (high * self._threshold)
                for high, close in zip(recent_closes[:-1], recent_closes[1:])
            )
            if is_continuing:
                continuation_symbols.append(symbol)

        continuation_symbols = continuation_symbols[:5]
        if not continuation_symbols:
            return Signal(information_available_at=stamp, weights={})
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
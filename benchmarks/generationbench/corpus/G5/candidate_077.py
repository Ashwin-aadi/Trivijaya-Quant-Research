from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Continuation breakouts occur after a strong move in one direction. "
        "A breakout followed by further movement in the same direction suggests strength and persistence."
    )

    def __init__(self, window: int = 30, threshold: float = 1.05) -> None:
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

            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(values) < self._window + 1:
                continue

            # Calculate the breakout condition
            last_close = values[-1]
            max_high = max(values[:-1])
            min_low = min(values[:-1])

            if (last_close > max_high and last_close >= max_high * self._threshold) or \
                    (last_close < min_low and last_close <= min_low / self._threshold):
                breakout_symbols.append(symbol)

        # Filter symbols that have continued in the same direction
        continuation_symbols: list[str] = []
        for symbol in breakout_symbols:
            history_slice = view.history(lookback=self._window)
            if symbol not in history_slice.columns:
                continue

            values = [float(v) for v in history_slice[symbol].drop_nulls().to_list()]
            last_close = float(history_slice["close"].max())
            direction = (values[-1] > last_close) - (values[-1] < last_close)

            if all((v[direction] >= 0 and values[i + 1] - v[direction] > 0 for i, v in enumerate(values[:-1]))):
                continuation_symbols.append(symbol)

        # Limit the number of symbols
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
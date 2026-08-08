from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a breakout, the stock often continues in the direction of the initial move. "
        "This strategy identifies stocks that have recently broken out and are likely to continue "
        "in that direction."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_sigs: dict[str, bool] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(values) < self._window + 1:
                continue

            close = values[-1]
            high_20d = max(values[:-1])
            low_20d = min(values[:-1])

            breakout_direction = (close - low_20d) / abs(high_20d - low_20d)
            if breakout_direction > self._threshold:
                breakout_sigs[symbol] = True
            elif breakout_direction < -self._threshold:
                breakout_sigs[symbol] = False

        continuation_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns or symbol not in breakout_sigs:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(values) < self._window + 1:
                continue

            close = values[-1]
            prev_close = values[-2]

            # Check for continuation in the breakout direction
            if (breakout_sigs[symbol] and close > prev_close) or (
                not breakout_sigs[symbol] and close < prev_close
            ):
                continuation_symbols.append(symbol)

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
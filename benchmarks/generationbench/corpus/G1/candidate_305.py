from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "After a breakout, if the price continues to move in the same direction for several days, "
        "it signals strong momentum and potential further increases. This strategy seeks to capitalize "
        "on such momentum."
    )

    def __init__(self, window: int = 20, continuation_days: int = 3) -> None:
        self._window = window
        self._continuation_days = continuation_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + self._continuation_days:
            return Signal(information_available_at=stamp, weights={})

        continuation_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window + self._continuation_days:
                continue

            # Check if the price continues to move in the same direction after the breakout window.
            breakout_direction = 1 if values[-1] > max(values[: self._window]) else -1
            continuation_prices = values[self._window : self._window + self._continuation_days]
            for i in range(1, len(continuation_prices)):
                if (continuation_prices[i] >= continuation_prices[0]) != (
                    breakout_direction == 1
                ):
                    break
            else:
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
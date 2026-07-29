from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts that continue to rise or fall over an extended period are often reliable "
        "signals of sustained momentum. This strategy identifies such breakouts by checking if a "
        "symbol's price continues in the breakout direction for several sessions."
    )

    def __init__(self, window: int = 20, continuation_period: int = 5) -> None:
        self._window = window
        self._continuation_period = continuation_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window + self._continuation_period)
        if closes.height < self._window + self._continuation_period:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window + self._continuation_period:
                continue

            breakout_price = max(values[: self._window])
            continuation_direction = 1.0 if values[self._window] > breakout_price else -1.0
            for i in range(self._window, self._window + self._continuation_period):
                if (values[i] - values[i - 1]) * continuation_direction <= 0:
                    break
            else:
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
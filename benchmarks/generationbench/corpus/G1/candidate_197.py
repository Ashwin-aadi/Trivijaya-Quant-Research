from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "After a breakout, stocks that maintain their momentum are likely to continue trending. "
        "This strategy identifies such stocks by looking for breakout continuation over a period."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window + self._lookback)
        if closes.height < self._window + self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window + self._lookback:
                continue

            breakout_date = max(
                range(len(values) - 1, self._window, -1),
                key=lambda x: values[x] >= max(values[:x]),
            )
            continuation = all(
                values[i] > values[i - 1]
                for i in range(breakout_date + 1, min(breakout_date + self._lookback, len(values)))
            )

            if breakout_date < len(values) - 1 and continuation:
                breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[:5]  # Top 5 symbols
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
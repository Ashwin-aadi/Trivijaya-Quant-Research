from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a price breakout, the continuation of the trend can often provide "
        "profitable opportunities. This strategy looks for symbols that have recently "
        "broken out and are likely to continue in their direction by using moving averages."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            last_close = values[-1]
            prev_close = values[-2]

            # Check if the price broke out either up or down
            breakout_condition = (last_close > max(values[:-1]) and prev_close <= max(values[:-1])) or \
                                 (last_close < min(values[:-1]) and prev_close >= min(values[:-1]))
            if not breakout_condition:
                continue

            # Validate continuation using a simple moving average
            avg_price = sum(values) / len(values)
            is_above_ma = last_close > avg_price and prev_close <= avg_price

            if is_above_ma:
                breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[: self._top_n]
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
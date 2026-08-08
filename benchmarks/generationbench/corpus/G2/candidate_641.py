from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakout continuation strategies aim to capitalize on the momentum that often follows "
        "a price breakout. After a security breaks above its recent resistance level, it is likely "
        "to continue moving upwards due to psychological and technical factors."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window + self._lookback:
                continue

            breakout_price = max(values[-self._window :])
            breakout_date = _find_broken_out_day(values, breakout_price)

            if breakout_date is not None and (
                values[breakout_date] == breakout_price
                or values[breakout_date + 1] > breakout_price
            ):
                continuation_prices = values[breakout_date + 1 : breakout_date + self._lookback + 1]
                if all(x >= y for x, y in zip(continuation_prices, values[breakout_date:])) and any(
                    x > y for x, y in zip(continuation_prices, values[breakout_date:])
                ):
                    breakout_symbols.append(symbol)

        weight = 1.0 / len(breakout_symbols)
        return Signal(information_available_at=stamp, weights={s: weight for s in breakout_symbols})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _find_broken_out_day(values: list[float], breakout_price: float) -> int | None:
    for i in range(len(values) - self._window):
        if values[i] < breakout_price and any(x >= breakout_price for x in values[i + 1 :]):
            return i
    return None
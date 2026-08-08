from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After identifying a breakout in the short term, continue to hold positions "
        "that show strong momentum during the subsequent period. This strategy aims "
        "to capture gains from stocks that have already shown strength."
    )

    def __init__(self, window_breakout: int = 20, window_momentum: int = 10) -> None:
        self._window_breakout = window_breakout
        self._window_momentum = window_momentum

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_breakout + self._window_momentum)

        if history.height < self._window_breakout + self._window_momentum:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue

            close_series = history.select(pl.col("symbol") == symbol)[
                "close"
            ].to_list()[0]

            if len(close_series) < self._window_breakout + self._window_momentum:
                continue

            breakout_price = close_series[-self._window_breakout]
            recent_close = close_series[-1]

            if recent_close > max(close_series[:-self._window_breakout]):
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        momentum_symbols: list[str] = []
        for symbol in breakout_symbols:
            history_slice = history.select(pl.col("symbol") == symbol)
            recent_prices = history_slice["close"].to_list()[0][-self._window_momentum:]

            if len(recent_prices) < self._window_momentum:
                continue

            momentum = sum([recent_prices[i] > recent_prices[i-1] for i in range(1, len(recent_prices))])
            if momentum >= (self._window_momentum - 1):
                momentum_symbols.append(symbol)

        weight = 1.0 / len(momentum_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in momentum_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakout continuation strategies exploit the phenomenon where stocks that have recently "
        "broken out of a consolidation pattern often continue in their breakout direction. This "
        "is based on the idea that breakout signals are not random and can be profitable if followed."
    )

    def __init__(self, window: int = 20, continuation_window: int = 10) -> None:
        self._window = window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_window)
        if history.is_empty() or history.height < self._window + self._continuation_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            history_for_symbol = history.filter(pl.col("symbol") == symbol)
            adj_close_series = history_for_symbol["adj_close"].to_list()
            if len(adj_close_series) < self._window + self._continuation_window:
                continue

            breakout_price = max(adj_close_series[-self._window:])
            last_day_price = adj_close_series[-1]

            if last_day_price > breakout_price:
                # Bullish breakout
                if all(last_day_price > adj_close >= breakout_price for adj_close in adj_close_series[-self._continuation_window:]):
                    breakout_symbols.append(symbol)
            elif last_day_price < breakout_price:
                # Bearish breakout
                if all(last_day_price < adj_close <= breakout_price for adj_close in adj_close_series[-self._continuation_window:]):
                    breakout_symbols.append(symbol)

        breakout_symbols = list(set(breakout_symbols))[: self._top_n(20)]
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
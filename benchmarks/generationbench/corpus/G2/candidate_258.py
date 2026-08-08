from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks that have outperformed relative to their peers in the broad universe over a "
        "recent period may continue to do so due to superior fundamentals or market sentiment."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or len(closes.columns) < 2:
            return Signal(information_available_at=stamp, weights={})

        symbol_returns = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            latest_close = float(view.latest_close()[symbol])
            base_period_close = values[0]
            return_periodic = (latest_close - base_period_close) / base_period_close
            symbol_returns[symbol] = return_periodic

        sorted_symbols = sorted(symbol_returns, key=symbol_returns.get, reverse=True)
        picks = sorted_symbols[: self._top_n]

        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
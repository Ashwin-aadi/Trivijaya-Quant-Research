from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in the direction of the breakout. This strategy "
        "identifies symbols that have recently broken out and continues to hold them."
    )

    def __init__(self, window: int = 20, lookback: int = 5) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback + self._window)

        if history.height < self._window + self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            symbol_history = history.select(
                pl.col("session_date"), pl.col(symbol)
            ).sort("session_date")

            if (
                (symbol_history["session_date"].to_list()[-1] - 
                 symbol_history["session_date"].to_list()[0]).days
                < self._window + self._lookback
            ):
                continue

            recent_high = float(
                max(symbol_history[symbol].drop_nulls().to_list()) / 
                symbol_history[symbol][-self._window:].mean()
            )
            if recent_high > 1.05:
                breakout_symbols.append(symbol)

        breakout_symbols = set(breakout_symbols)
        weight = 1.0 / len(breakout_symbols) if breakout_symbols else 0
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in breakout_symbols},
            actions="buy_and_hold"
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
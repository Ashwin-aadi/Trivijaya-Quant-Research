from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeTrendBreakout(Strategy):
    rationale = (
        "This strategy seeks to capitalize on both short-term momentum and medium-term trend "
        "breakouts. Short-term momentum can indicate an asset's recent performance strength, "
        "while a medium-term breakout suggests a potential continuation of the trend."
    )

    def __init__(self, short_window: int = 5, long_window: int = 20) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._long_window)
        if closes.height < self._long_window:
            return Signal(information_available_at=stamp, weights={})

        short_returns = (closes[closes.columns[1]] / closes[closes.columns[0]].shift(1) - 1.0).to_list()
        long_returns = (closes[closes.columns[1]] / closes[closes.columns[0]].shift(self._short_window) - 1.0).to_list()

        if len(short_returns) < self._short_window or len(long_returns) < self._long_window:
            return Signal(information_available_at=stamp, weights={})

        short_mean = sum(short_returns[-self._short_window:]) / self._short_window
        long_mean = sum(long_returns[-self._long_window:]) / self._long_window

        if short_mean > 0.1 and long_mean > 0.05:
            picks: list[str] = []
            for symbol in view.symbols:
                if symbol not in closes.columns:
                    continue
                values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
                if values[-1] >= max(values):
                    picks.append(symbol)

            weight = 1.0 / len(picks)
            return Signal(
                information_available_at=stamp, weights={s: weight for s in picks}
            )
        else:
            return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
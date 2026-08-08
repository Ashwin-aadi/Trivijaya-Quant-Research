from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy combines a breakout signal with a momentum indicator to capture "
        "both short-term reversals and long-term trends."
    )

    def __init__(self, window_breakout: int = 20, top_n: int = 5, momentum_window: int = 30) -> None:
        self._window_breakout = window_breakout
        self._top_n = top_n
        self._momentum_window = momentum_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_breakout + self._momentum_window)

        if history.height < self._window_breakout + self._momentum_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_picks: list[str] = []
        momentum_picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_series = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(close_series) < self._window_breakout + self._momentum_window:
                continue

            breakout_condition = (
                close_series[-1] >= max(close_series[-self._window_breakout:])
                and close_series[-2] < max(close_series[-self._window_breakout - 1 : -1])
            )
            if not breakout_condition:
                continue

            breakout_picks.append(symbol)

            momentum_condition = (
                (close_series[-1] / close_series[0]) > (1 + 0.02)
                and (close_series[-1] - close_series[0]) > (0.05 * close_series[0])
            )
            if not momentum_condition:
                continue

            momentum_picks.append(symbol)

        combined_picks = list(set(breakout_picks) & set(momentum_picks))
        if not combined_picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(combined_picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in combined_picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
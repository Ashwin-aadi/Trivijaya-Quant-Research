from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakout continuation strategies look for stocks that have recently broken out of a "
        "range and are trending. These stocks often continue their trend after the initial breakout."
    )

    def __init__(self, window: int = 20, breakout_window: int = 10) -> None:
        self._window = window
        self._breakout_window = breakout_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._breakout_window)

        if history.height < self._window + self._breakout_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = set()
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            hist_data = history[symbol].drop_nulls().to_list()
            if len(hist_data) < self._window + self._breakout_window:
                continue

            # Calculate daily returns
            daily_returns = [
                float(hist_data[i + 1] / hist_data[i] - 1.0)
                for i in range(len(hist_data) - 1)
            ]
            
            # Find breakout points
            for i in range(self._breakout_window, len(daily_returns)):
                if (
                    daily_returns[i - self._breakout_window: i]
                    .max()
                    .item() == daily_returns[i - self._breakout_window]
                ):
                    breakout_symbols.add(symbol)
                    break

        # Filter out symbols that do not have a valid breakout
        breakout_symbols = [s for s in breakout_symbols if s in view.symbols]

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
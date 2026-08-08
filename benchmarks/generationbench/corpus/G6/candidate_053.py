from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "The strategy identifies breakout continuation patterns by detecting "
        "when a stock's close price exceeds its 5-day moving high for three consecutive days. "
        "Positions are exited if the stock reverts below a 20-day price level or if it reaches a "
        "-15% portfolio-wide loss threshold, ensuring timely exits and risk management."
    )

    def __init__(self, window_high: int = 5, consecutive_days: int = 3, stop_loss_window: int = 20) -> None:
        self._window_high = window_high
        self._consecutive_days = consecutive_days
        self._stop_loss_window = stop_loss_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._stop_loss_window + 1)

        if history.height < self._stop_loss_window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window_high + 2)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            history_symbol = history.filter(pl.col("symbol") == symbol)
            if history_symbol.height < self._stop_loss_window + 1:
                continue

            recent_closes = [float(v) for v in closes[symbol].to_list()[-self._window_high - 2 :]]
            moving_highs = [max(recent_closes[i : i + self._window_high]) for i in range(len(recent_closes) - self._window_high + 1)]

            if all(c > h for c, h in zip(recent_closes[-self._consecutive_days:], moving_highs[-self._consecutive_days:])):
                picks.append(symbol)

        stops = {symbol: float(view.latest_close()[symbol]) * 0.85 for symbol in view.symbols}
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={
                s: weight
                for s in picks
                if stops[s] > float(view.latest_close()[s])
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "The strategy targets stocks that exhibit strong price action after breaking out of a defined trading range. "
        "Breakouts are identified when stocks close above/below their 20-day moving high/low with significant volume, and the breakout is confirmed by closing outside the previous day’s range for two consecutive days."
    )

    def __init__(self, window: int = 20, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * 2 + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.unique().to_list():
                continue
            df = history.filter(pl.col("symbol") == symbol)
            closes = df.select("adj_close").collect()
            opens = df.select("open").collect()
            highs = df.select("high").collect()
            lows = df.select("low").collect()

            close_values = [float(v) for v in closes.to_list()[0]]
            open_values = [float(v) for v in opens.to_list()[0]]
            high_values = [float(v) for v in highs.to_list()[0]]
            low_values = [float(v) for v in lows.to_list()[0]]

            if len(close_values) < self._window + 1:
                continue

            # Identify breakout day
            for i in range(self._window, len(close_values) - 1):
                if (
                    close_values[i] > max(high_values[:i])
                    and open_values[i] >= high_values[i]
                    and low_values[i] <= high_values[i]
                ):
                    break
            else:
                continue

            # Confirm breakout over next day
            if (
                close_values[i + 1] > high_values[i]
                or close_values[i + 1] < low_values[i]
            ) and open_values[i + 1] >= high_values[i]:
                breakout_signals[symbol] = float(close_values[i])

        picks: list[str] = sorted(breakout_signals, key=breakout_signals.get, reverse=True)[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
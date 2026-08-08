from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related signals: the 20-day momentum and the 50-day moving average "
        "crossover. The combination aims to capture both short-term trends and longer-term support/resistance levels."
    )

    def __init__(self, window_20d: int = 20, window_50d: int = 50) -> None:
        self._window_20d = window_20d
        self._window_50d = window_50d

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=max(self._window_20d, self._window_50d))
        if closes.height < max(self._window_20d, self._window_50d):
            return Signal(information_available_at=stamp, weights={})

        twenty_day_moments: list[float] = []
        fifty_day_signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window_20d:
                continue

            twenty_day_moments.append(
                (values[-1] / values[-self._window_20d - 1]) - 1.0
            )

            if len(values) < self._window_50d:
                fifty_day_signals.append("neutral")
                continue

            avg_50 = sum(values[-self._window_50d : -1]) / (self._window_50d - 1)
            if values[-1] > avg_50:
                fifty_day_signals.append("bullish")
            else:
                fifty_day_signals.append("bearish")

        top_20 = sorted(twenty_day_moments, reverse=True)[:3]
        bullish_count = sum(1 for signal in fifty_day_signals if signal == "bullish")

        picks: list[str] = []
        weight = 1.0 / len(top_20) * (0.6 + 0.4 * bullish_count / self._window_50d)
        for symbol, moment in zip(view.symbols, twenty_day_moments):
            if moment in top_20:
                picks.append(symbol)

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
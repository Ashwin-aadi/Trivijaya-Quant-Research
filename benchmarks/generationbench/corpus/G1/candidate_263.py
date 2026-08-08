from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "Combining short-term momentum with long-term trend following can capture both "
        "immediate price action and broader market trends, potentially reducing whipsaws."
    )

    def __init__(self, short_window: int = 10, long_window: int = 60) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._short_window, self._long_window))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes_short = (
            view.closes(lookback=self._short_window).sort("session_date").select(
                pl.col(view.symbols[0]).alias("close")
            )
        )
        closes_long = (
            view.closes(lookback=self._long_window).sort("session_date").select(
                pl.col(view.symbols[0]).alias("close")
            )
        )

        short_moments: list[float] = []
        long_moments: list[float] = []

        for symbol in view.symbols:
            if symbol not in closes_short.columns or symbol not in closes_long.columns:
                continue
            short_close_values = [float(v) for v in closes_short[symbol].to_list()]
            long_close_values = [float(v) for v in closes_long[symbol].to_list()]

            if len(short_close_values) < self._short_window or len(long_close_values) < self._long_window:
                continue

            short_moment = (
                (short_close_values[-1] / short_close_values[0]) - 1.0
            )
            long_moment = (
                (long_close_values[-1] / long_close_values[0]) - 1.0
            )

            short_moments.append(short_moment)
            long_moments.append(long_moment)

        if not short_moments or not long_moments:
            return Signal(information_available_at=stamp, weights={})

        combined_moments = [
            (short * 0.6) + (long * 0.4) for short, long in zip(short_moments, long_moments)
        ]

        sorted_indices = [i[0] for i in reversed(sorted(enumerate(combined_moments), key=lambda x: x[1]))]
        top_symbols = [view.symbols[i] for i in sorted_indices[:5]]

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
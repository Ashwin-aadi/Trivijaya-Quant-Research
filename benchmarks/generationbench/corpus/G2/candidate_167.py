from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeTrendBreakout(Strategy):
    rationale = (
        "We hypothesize that stocks which have recently broken out of their long-term trends "
        "are likely to continue in the direction of their recent momentum. This strategy aims "
        "to identify such stocks by combining a short-term breakout signal with a longer-term "
        "trend change."
    )

    def __init__(self, short_window: int = 20, long_window: int = 100) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._short_window + self._long_window - 1)
        if closes.height < self._short_window + self._long_window - 1:
            return Signal(information_available_at=stamp, weights={})

        short_breakouts: list[str] = []
        long_trends: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue

            short_prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(short_prices) < self._short_window + self._long_window - 1:
                continue

            # Check for a breakout
            if (
                (short_prices[-1] > max(short_prices[-self._short_window :]))
                and short_prices[-20:] != sorted(short_prices[-20:])
            ):
                short_breakouts.append(symbol)

            # Calculate long-term trend change
            long_trend = float(
                (short_prices[-1] - short_prices[0])
                / sum(short_prices[: self._long_window]) * 100.0
            )
            if abs(long_trend) > 2.0:
                long_trends[symbol] = long_trend

        # Select symbols that have both a recent breakout and a significant long-term trend change
        picks: list[str] = [
            symbol for symbol in short_breakouts if long_trends.get(symbol, 0.0) != 0.0
        ]

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
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "This strategy combines short-term and long-term momentum indicators to identify "
        "overbought or oversold conditions. By leveraging both perspectives, it aims to capture "
        "potentially profitable opportunities more accurately."
    )

    def __init__(self, short_window: int = 10, long_window: int = 60) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._short_window + self._long_window)
        if closes.height < self._short_window + self._long_window:
            return Signal(information_available_at=stamp, weights={})

        short_moments: list[str] = []
        long_moments: list[str] = []

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._short_window + self._long_window:
                continue

            short_return = (values[-1] / values[-self._short_window - 1] - 1.0)
            long_return = (values[-1] / values[-self._long_window - 1] - 1.0)

            if short_return > 0 and long_return > 0:
                short_moments.append(symbol)
            elif short_return < 0 and long_return < 0:
                long_moments.append(symbol)

        combined = set(short_moments) & set(long_moments)
        if not combined:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(combined)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in combined}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
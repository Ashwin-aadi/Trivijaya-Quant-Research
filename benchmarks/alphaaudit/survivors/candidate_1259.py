from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: the 20-day high and the "
        "50-day low. The idea is that stocks hitting their recent highs and lows could be "
        "indicative of potential reversals or continuation patterns."
    )

    def __init__(self, high_window: int = 20, low_window: int = 50) -> None:
        self._high_window = high_window
        self._low_window = low_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._high_window + self._low_window)
        if closes.height < self._high_window + self._low_window:
            return Signal(information_available_at=stamp, weights={})

        high_symbols: list[str] = []
        low_symbols: list[str] = []

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._high_window + self._low_window:
                continue

            high_value = max(values[-self._high_window:])
            low_value = min(values[-self._low_window:])

            if values[-1] == high_value:
                high_symbols.append(symbol)
            elif values[-1] == low_value:
                low_symbols.append(symbol)

        top_highs = high_symbols[:5]
        bottom_lows = low_symbols[:5]

        picks = list(set(top_highs + bottom_lows))
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
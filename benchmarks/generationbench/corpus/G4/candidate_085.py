from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion520(Strategy):
    rationale = (
        "This strategy exploits short-horizon mean reversion by identifying stocks "
        "where the 5-day moving average crosses below the 20-day moving average, "
        "suggesting a potential upward trend reversal."
    )

    def __init__(self, short_window: int = 5, long_window: int = 20) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._long_window + 1)
        if closes.height < self._long_window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            short_ma = sum(prices[-self._short_window:]) / self._short_window
            long_ma = sum(prices[-self._long_window:]) / self._long_window
            if short_ma < long_ma:
                symbols.append(symbol)

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
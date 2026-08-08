from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that assets which have moved significantly from their mean "
        "trend value will revert back towards it. For short horizons (such as 10 days), this "
        "can provide trading opportunities."
    )

    def __init__(self, window: int = 10, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or not all(
            col in view.symbols for col in closes.columns
        ):
            return Signal(information_available_at=stamp, weights={})

        symbol_values: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            mean_price = sum(values[-self._window:]) / self._window
            latest_close = values[-1]
            deviation = abs(latest_close - mean_price)
            if deviation > self._threshold * mean_price:
                symbol_values.append((symbol, deviation))

        sorted_symbol_values = sorted(symbol_values, key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in sorted_symbol_values[:5]]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

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
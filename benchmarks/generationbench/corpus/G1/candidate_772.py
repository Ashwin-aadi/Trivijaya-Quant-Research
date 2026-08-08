from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price reversion occurs when a stock's price returns to its mean after deviating from it. "
        "By tracking the trailing average, we can identify overbought or oversold conditions and "
        "generate buy/sell signals accordingly."
    )

    def __init__(self, window: int = 50, trailing_window: int = 20) -> None:
        self._window = window
        self._trailing_window = trailing_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            mean_price = sum(values[-self._trailing_window:]) / self._trailing_window
            latest_close = values[-1]
            if latest_close > (mean_price * 1.05):
                signals[symbol] = -0.1
            elif latest_close < (mean_price * 0.95):
                signals[symbol] = 0.1

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(signals.values())
        adjusted_signals = {s: w / total_weight for s, w in signals.items()}
        return Signal(
            information_available_at=stamp, weights=adjusted_signals
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
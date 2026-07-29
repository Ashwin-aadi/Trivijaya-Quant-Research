from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Certain stocks exhibit seasonal behavior due to industry-specific factors or calendar effects. "
        "By identifying and exploiting these patterns, we can generate alpha in the market."
    )

    def __init__(self, window: int = 365) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_signals = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate the average close and the current value
            avg_close = sum(values) / self._window
            current_close = values[-1]

            # Check if the current close is above or below the seasonal average
            if current_close > avg_close * 1.05:  # Threshold for buy signal
                seasonal_signals[symbol] = "buy"
            elif current_close < avg_close * 0.95:  # Threshold for sell signal
                seasonal_signals[symbol] = "sell"

        weights = {s: 0.0 for s in view.symbols}
        if seasonal_signals:
            buy_symbols = [symbol for symbol, signal in seasonal_signals.items() if signal == "buy"]
            sell_symbols = [symbol for symbol, signal in seasonal_signals.items() if signal == "sell"]

            # Allocate weight to buy symbols
            if buy_symbols:
                weights.update({s: 1.0 / len(buy_symbols) for s in buy_symbols})

            # Allocate weight to sell symbols (as cash)
            if sell_symbols:
                weights[sell_symbols[0]] = -1.0

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
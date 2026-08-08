from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "High volatility periods often precede significant market movements. By identifying "
        "symbols that have recently experienced high price swings, we can capture trends in "
        "the early stages."
    )

    def __init__(self, window: int = 20, threshold_multiplier: float = 1.5) -> None:
        self._window = window
        self._threshold_multiplier = threshold_multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_volatility: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            price_changes = [close_prices[i] - close_prices[i-1] for i in range(1, len(close_prices))]
            volatility = abs(max(price_changes)) / min(price_changes)
            if volatility > self._threshold_multiplier:
                symbol_volatility[symbol] = volatility

        selected_symbols = sorted(symbol_volatility.items(), key=lambda x: x[1], reverse=True)[:5]
        weights = {symbol: 1.0 for _, symbol in selected_symbols}

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weights.items() if w > 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
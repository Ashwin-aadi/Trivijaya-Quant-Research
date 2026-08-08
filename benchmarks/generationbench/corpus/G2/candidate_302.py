from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySCTF(Strategy):
    rationale = (
        "Volatility-scaled trend-following strategies attempt to capture the momentum of "
        "trending markets while adjusting position sizes based on recent volatility. During"
        " periods of high volatility, smaller positions are taken to limit risk; during low"
        " volatility, larger positions can be taken."
    )

    def __init__(self, window: int = 60, volatility_window: int = 30, threshold: float = 1.5) -> None:
        self._window = window
        self._volatility_window = volatility_window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        close_series = [float(v) for v in history["adj_close"].to_list()]
        returns = [(close_series[i] - close_series[i-1]) / close_series[i-1] for i in range(1, len(close_series))]

        # Calculate volatility over the last `volatility_window` days
        vol_history = view.history(lookback=self._volatility_window)
        if vol_history.height < self._volatility_window:
            return Signal(information_available_at=stamp, weights={})

        close_values = [float(v) for v in vol_history["adj_close"].drop_nulls().to_list()]
        daily_returns = [(close_values[i] - close_values[i-1]) / close_values[i-1] for i in range(1, len(close_values))]
        volatility = max([abs(r) for r in daily_returns])

        # Determine the number of symbols to trade based on volatility
        if volatility < self._threshold:
            n_symbols = 5
        else:
            n_symbols = 2

        # Select symbols with positive returns
        symbol_returns = [returns[i-1] for i, s in enumerate(history["symbol"].to_list()) if s in view.symbols]
        top_symbols = sorted(zip(view.symbols, symbol_returns), key=lambda x: x[1], reverse=True)[:n_symbols]

        weights = {s: 1.0 / n_symbols for s, _ in top_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy seeks to combine trend-following and value metrics. The idea is that "
        "companies with high return on assets (ROA) tend to have positive trends in their stock prices, "
        "which can be identified through recent price movements. By combining these two factors, we aim "
        "to capture both the strength of the current trend and the underlying value."
    )

    def __init__(self, roa_window: int = 30, trend_window: int = 20) -> None:
        self._roa_window = roa_window
        self._trend_window = trend_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._trend_window)
        if closes.height < self._trend_window:
            return Signal(information_available_at=stamp, weights={})

        roa = view.closes(lookback=self._roa_window)
        if roa.height < self._roa_window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate ROA
        roa_values = {symbol: float(roa[symbol].drop_nulls().to_list()[-1]) for symbol in view.symbols}
        
        # Calculate trend strength
        trends = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._trend_window:
                continue
            last_close = values[-1]
            mean_last_n_days = sum(values[-self._trend_window:]) / self._trend_window
            trend_strength = (last_close - mean_last_n_days) / mean_last_n_days * 100.0
            trends[symbol] = trend_strength

        # Filter symbols based on both criteria
        valid_symbols = [symbol for symbol in view.symbols if symbol in roa_values and symbol in trends]
        if not valid_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Assign equal weight to all selected stocks
        weight = 1.0 / len(valid_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in valid_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
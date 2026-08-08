from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends adjusted by historical volatility. The 30-day moving "
        "average of daily high minus daily low (range) is scaled by the 75-day historical "
        "volatility to capture both intraday price action and trend strength, while overall "
        "market volatility adjusts the signal."
    )

    def __init__(self, window_range: int = 30, window_volatility: int = 75, top_n: int = 5) -> None:
        self._window_range = window_range
        self._window_volatility = window_volatility
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_range + self._window_volatility)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_history = {}
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            highs = [float(v) for v in df["high"].to_list()]
            lows = [float(v) for v in df["low"].to_list()]
            if len(highs) < self._window_range or len(lows) < self._window_volatility:
                continue
            range_values = [(h - l) / 2 for h, l in zip(highs[-self._window_range:], lows[-self._window_range:])]
            vol_values = [float(v) for v in df["adj_close"].shift(-1).to_list()[:-1]]
            if len(range_values) < self._window_range or len(vol_values) < self._window_volatility:
                continue
            range_mean = sum(range_values[-self._window_range:]) / self._window_range
            vol_mean = (sum([(v - sum(vol_values[-self._window_volatility:]) / self._window_volatility)**2 for v in vol_values[-self._window_volatility:]])
                        / self._window_volatility) ** 0.5
            if vol_mean == 0:
                continue
            scaled_range = range_mean / (vol_mean * 10)
            symbol_history[symbol] = scaled_range

        symbols_sorted = sorted(symbol_history.items(), key=lambda x: x[1], reverse=True)[:self._top_n]
        weights = {s: 0.2 for s, _ in symbols_sorted}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
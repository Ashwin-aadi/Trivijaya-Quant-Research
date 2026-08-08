from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends by identifying symbols that have been trending up or down "
        "for a significant period. It uses volatility to scale the trend signal, ensuring that "
        "high-volatility assets are entered with smaller weights compared to low-volatility ones."
    )

    def __init__(self, window: int = 20, min_periods: int = 15, vol_window: int = 30) -> None:
        self._window = window
        self._min_periods = min_periods
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._vol_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        symbols = history["symbol"].to_list()

        trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in symbols:
                continue
            index = symbols.index(symbol)
            close_series = [float(c) for c in closes[index : index + self._window]]
            vol_series = [
                float(c)
                for c in history["adj_close"].to_list()[index : index + self._vol_window]
            ]
            if len(close_series) < self._window or len(vol_series) < self._vol_window:
                continue

            trend = close_series[-1] - close_series[0]
            vol = pl.Series(vol_series).std()
            trends[symbol] = (trend / vol)

        top_symbols = sorted(trends.items(), key=lambda x: abs(x[1]), reverse=True)[: self._min_periods]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight * abs(v) for s, v in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
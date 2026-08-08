from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrend(Strategy):
    rationale = (
        "This strategy identifies symbols with a recent trend in volatility. A rising trend "
        "in volatility often precedes price reversals or significant moves. By identifying such "
        "symbols, we aim to capture potential turning points in the market."
    )

    def __init__(self, window1: int = 20, window2: int = 60) -> None:
        self._window1 = window1
        self._window2 = window2

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window2)

        if history.height < self._window2:
            return Signal(information_available_at=stamp, weights={})

        symbols_with_trend = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            open_prices = [float(v) for v in history[symbol][["open"]].to_list()]
            adj_closes = [float(v) for v in history[symbol][["adj_close"]].to_list()]

            vol1 = _rolling_std(adj_closes, window=self._window1)
            vol2 = _rolling_std(adj_closes, window=self._window2)

            if len(vol1) < self._window2:
                continue

            trend = [vol1[i] <= vol2[i] for i in range(len(vol1))]
            if all(trend):
                symbols_with_trend.append(symbol)

        weight = 1.0 / len(symbols_with_trend)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in symbols_with_trend}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _rolling_std(adj_closes: list[float], window: int) -> list[float]:
    stds = []
    for i in range(len(adj_closes) - window + 1):
        subset = adj_closes[i : i + window]
        mean = sum(subset) / len(subset)
        variance = sum((x - mean) ** 2 for x in subset) / (len(subset) - 1)
        stds.append(variance**0.5)
    return stds
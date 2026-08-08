from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy leverages the inverse relationship between volatility and trending "
        "behavior in stocks. During low-volatility periods, larger positions are taken to "
        "capitalize on extended trends, while higher-volatility periods see smaller position "
        "sizes to mitigate risk."
    )

    def __init__(self, window: int = 20, max_positions: int = 30) -> None:
        self._window = window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window * 2 + 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        open_prices = [float(v) for v in history["open"].to_list()]
        high_prices = [float(v) for v in history["high"].to_list()]
        low_prices = [float(v) for v in history["low"].to_list()]
        close_prices = [float(v) for v in history["close"].to_list()]

        volatility = [
            (max(h, l) - min(h, l)) / c
            for o, h, l, c in zip(open_prices[1:], high_prices[1:], low_prices[1:], close_prices[1:])
        ]

        vol_ma = sum(volatility[-self._window:]) / self._window

        weights: dict[str, float] = {}
        positions = 0
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_close_series = [float(v) for v in history[symbol].to_list()]
            if len(adj_close_series) < self._window * 2 + 1:
                continue

            recent_adj_closes = adj_close_series[-self._window:]
            recent_volatility = [(max(h, l) - min(h, l)) / c for h, l, c in zip(high_prices[-self._window:], low_prices[-self._window:], close_prices[-self._window:])]

            scaling_factor = 1.0 / (vol_ma + 1e-8)
            weight = scaling_factor * len(recent_adj_closes) / self._max_positions
            weights[symbol] = min(5.0, max(weight, 2.0)) / vol_ma

            positions += 1
            if positions >= self._max_positions:
                break

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion seeks to exploit deviations from the mean price. "
        "When a stock's price moves away from its historical average, it is expected to revert back."
    )

    def __init__(self, window: int = 10, z_score_threshold: float = 2.0) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or history.width == 0:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        closes = history[symbols].select("adj_close").to_pandas().set_index("session_date")
        means = closes.resample("D").mean()
        z_scores = (closes - means).rolling(window=self._window, min_periods=1).std().fillna(0)
        signals = ((z_scores < -self._z_score_threshold) | (z_scores > self._z_score_threshold)).to_dict()

        weights: dict[str, float] = {}
        for symbol, signal in signals.items():
            if any(signal):
                weights[symbol] = 1.0 / sum(signal)
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_datetime().date()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Short-horizon mean reversion strategy capitalizes on stock prices reverting to their historical averages. "
        "The entry rule identifies stocks that are significantly deviated from a 20-day simple moving average (SMA)."
    )

    def __init__(self, window: int = 20, z_score_threshold: float = -2, max_positions: int = 30) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        sma = sum(closes[-self._window:]) / self._window
        std_dev = (sum((c - sma) ** 2 for c in closes[-self._window:]) / self._window) ** 0.5

        z_scores = [(c - sma) / std_dev for c in closes]
        symbols = [symbol for symbol, z in zip(view.symbols, z_scores) if z < self._z_score_threshold]

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        selected_symbols = symbols[: self._max_positions]
        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
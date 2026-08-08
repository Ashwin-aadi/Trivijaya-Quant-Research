from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines two characteristics: the momentum of a stock and its "
        "volume anomaly. Momentum captures recent price movement, while volume anomaly "
        "helps identify potential significant trading activity."
    )

    def __init__(self, momentum_window: int = 10, volume_threshold: float = 2.0) -> None:
        self._momentum_window = momentum_window
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=2 * self._momentum_window + 10)
        if history.height < 2 * self._momentum_window + 10:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        picks: list[str] = []

        for symbol in symbols:
            if symbol not in history.columns:
                continue

            close_data = [float(v) for v in history[symbol].to_list()]
            momentum_score = (sum(close_data[-self._momentum_window:]) - sum(
                close_data[:-self._momentum_window])) / self._momentum_window

            volume_history = view.history(lookback=self._momentum_window)
            if symbol not in volume_history.columns:
                continue
            volumes = [float(v) for v in volume_history[symbol].to_list()]
            recent_volume_mean = sum([v > 0 for v in volumes[-self._momentum_window:]])

            if momentum_score > 0 and recent_volume_mean >= self._volume_threshold:
                picks.append(symbol)

        picks = picks[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMove(Strategy):
    rationale = (
        "This strategy identifies days where both price and volume show a clear directional move. "
        "Entering long or short positions based on these moves capitalizes on the market's tendency "
        "to validate and amplify trends through increased trading activity."
    )

    def __init__(self, bullish_threshold: float = 0.02, bearish_threshold: float = -0.02, lookback_days: int = 30) -> None:
        self._bullish_threshold = bullish_threshold
        self._bearish_threshold = bearish_threshold
        self._lookback_days = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns or len(history[symbol].to_list()) < self._lookback_days + 1:
                continue
            close_values = [float(v) for v in history[symbol][-self._lookback_days:].select("close").to_list()[0]]
            volume_values = [float(v) for v in history[symbol][-self._lookback_days:].select("volume").to_list()[0]]
            prev_close = float(close_values[-2])
            current_close = float(close_values[-1])

            bullish_signal = (current_close - prev_close) / prev_close * 100 > self._bullish_threshold
            bearish_signal = (prev_close - current_close) / prev_close * 100 > abs(self._bearish_threshold)

            if bullish_signal and volume_values[-1] > sum(volume_values) / len(volume_values):
                picks[symbol] = 1.0 / len(picks)
            elif bearish_signal and volume_values[-1] > sum(volume_values) / len(volume_values):
                picks[symbol] = -1.0 / len(picks)

        return Signal(information_available_at=stamp, weights=picks)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
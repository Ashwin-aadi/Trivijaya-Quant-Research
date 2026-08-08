from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirm(Strategy):
    rationale = (
        "Volume-confirmed directional moves are likely to continue. By identifying symbols that "
        "show significant volume on a breakout or break-in, we can capture momentum in the direction of the move."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            hist = history[history["symbol"] == symbol]
            close_prices = [float(v) for v in hist.select("adj_close").to_series().to_list()[0]]
            volumes = [float(v) for v in hist.select("volume").to_series().to_list()[0]]

            if len(close_prices) < self._window + 1:
                continue

            last_close = close_prices[-1]
            prev_close = close_prices[-2]
            volume_change = volumes[-1] - volumes[-2]

            if (last_close > prev_close and volume_change > 0) or (
                last_close < prev_close and volume_change < 0
            ):
                picks.append(symbol)

        picks = list(set(picks))[:5]
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
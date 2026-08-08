from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of strong market sentiment. "
        "A significant increase in volume on a price move suggests genuine buying or selling pressure."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        closes = [float(v) for v in history[symbol].drop_nulls().to_list() for symbol in symbols]
        volumes = [float(v) for v in history["volume"].drop_nulls().to_list()]

        candidates: list[str] = []
        for i in range(len(closes) - 1):
            current_close = closes[i]
            next_close = closes[i + 1]
            volume_change = (volumes[i + 1] - volumes[i]) / volumes[i]

            if (
                next_close > current_close
                and volume_change > 0.5
                or next_close < current_close
                and volume_change < -0.5
            ):
                candidates.append(symbols[i])

        weight = 1.0 / len(candidates) if candidates else 0.0
        return Signal(
            information_available_at=stamp, weights={s: weight for s in candidates}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
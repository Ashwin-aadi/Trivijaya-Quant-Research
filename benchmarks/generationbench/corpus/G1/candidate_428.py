from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment. By identifying "
        "sudden increases in volume alongside price movements, we can capture opportunities from "
        "significant market shifts."
    )

    def __init__(self, window: int = 10, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 2)
        if history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            hist = history.filter(pl.col("symbol") == symbol).sort("session_date")
            closes = [float(v) for v in hist["adj_close"].drop_nulls().to_list()]
            volumes = [float(v) for v in hist["volume"].drop_nulls().to_list()]

            if len(closes) < self._window + 2 or len(volumes) < self._window + 2:
                continue

            recent_close = closes[-1]
            recent_volume = volumes[-1]

            prev_close = closes[-(self._window + 1)]
            prev_volume = volumes[-(self._window + 1)]

            direction = (recent_close - prev_close) / abs(prev_close)
            vol_ratio = recent_volume / prev_volume

            if vol_ratio >= self._threshold and direction > 0:
                signals[symbol] = 1.0
            elif vol_ratio <= 1 / self._threshold and direction < 0:
                signals[symbol] = -1.0

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in signals.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
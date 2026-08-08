from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are more likely to be sustained trends. By "
        "identifying such moves, we can potentially capitalize on the continuation of these "
        "trends."
    )

    def __init__(self, window: int = 10, min_volume_ratio: float = 1.5) -> None:
        self._window = window
        self._min_volume_ratio = min_volume_ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns or "volume" not in history.columns:
                continue

            prices = [float(v) for v in history[symbol].to_list()]
            volumes = [float(v) for v in history["volume"].to_list()]

            direction = 1
            prev_price = None
            for i in range(1, len(prices)):
                if prev_price is not None and prices[i] > prices[i - 1]:
                    current_direction = 1
                elif prev_price is not None and prices[i] < prices[i - 1]:
                    current_direction = -1
                else:
                    continue

                if direction == current_direction and volumes[i] / volumes[i - 1] >= self._min_volume_ratio:
                    picks.append(symbol)
                    break

            prev_price = prices[-2]

        weight = 1.0 / len(picks) if picks else 0.0
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
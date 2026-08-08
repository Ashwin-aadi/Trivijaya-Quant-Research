from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Large volume movements in a single direction are often indicative of strong market "
        "sentiment and can lead to continuation of the trend. By identifying such moves, we aim "
        "to capture potentially profitable opportunities."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 3)

        if history.height < self._window + 3:
            return Signal(information_available_at=stamp, weights={})

        volume_changes = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            open_prices = [float(v) for v in history[symbol]["open"].to_list()]
            close_prices = [float(v) for v in history[symbol]["close"].to_list()]
            volumes = [float(v) for v in history[symbol]["volume"].to_list()]

            if len(open_prices) < self._window + 3:
                continue

            recent_open = open_prices[-1]
            recent_close = close_prices[-1]

            if (recent_open - min(close_prices)) > (max(open_prices) - recent_close):
                direction = "up"
            elif (recent_open - max(close_prices)) < (min(open_prices) - recent_close):
                direction = "down"
            else:
                continue

            total_volume = sum(volumes)
            recent_volume = volumes[-1]
            if recent_volume >= 0.5 * total_volume:  # Assume 50% of average volume
                volume_changes.append((symbol, direction))

        picks = [s[0] for s in volume_changes[:5]]
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
from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume surges on a price break beyond the recent range suggest strong conviction "
        "and may indicate continuation of the move. This strategy captures such moments."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol]["adj_close"].to_list()]
            volumes = [float(v) for v in history[symbol]["volume"].to_list()]

            # Calculate price change and volume
            changes = [adj_closes[i] - adj_closes[i-1] for i in range(1, len(adj_closes))]
            volumes_shifted = volumes[1:]

            # Find the break point where price has moved beyond its recent range
            last_close = adj_closes[-1]
            low_range = min(adj_closes)
            high_range = max(adj_closes)

            if last_close > high_range:
                breakout_price = "high"
                base_value = high_range
            elif last_close < low_range:
                breakout_price = "low"
                base_value = low_range
            else:
                continue

            # Check for volume surge on the break point
            if volumes[-1] > self._threshold * max(volumes_shifted):
                signals[symbol] = 1.0 / len(signals)

        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest